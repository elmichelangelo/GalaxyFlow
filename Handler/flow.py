import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).double()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)
            self.cond_linear.weight.data = self.cond_linear.weight.data.to(dtype=torch.float64)

        self.register_buffer('mask', mask.to(dtype=torch.float64))
        torch.set_default_dtype(torch.float64)

    def forward(self, inputs, cond_inputs=None):
        # if inputs.dtype is torch.float64:
        #     inputs = inputs.float()
        inputs = inputs.to(dtype=torch.float64)
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(dtype=torch.float64)
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 lrelu=None):
        super(MADE, self).__init__()

        activations = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'leaky_relu': nn.LeakyReLU
        }
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs,
            num_hidden,
            num_inputs,
            mask_type='input'
        )
        hidden_mask = get_mask(
            num_hidden,
            num_hidden,
            num_inputs
        )
        output_mask = get_mask(
            num_hidden,
            num_inputs * 2,
            num_inputs,
            mask_type='output'
        )

        self.joiner = nn.MaskedLinear(
            num_inputs,
            num_hidden,
            input_mask,
            num_cond_inputs
        )

        self.trunk = nn.Sequential(
            act_func(),
            nn.MaskedLinear(
                num_hidden,
                num_hidden,
                hidden_mask
            ),
            act_func(),
            nn.MaskedLinear(
                num_hidden,
                num_inputs * 2,
                output_mask
            )
        )

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs, dtype=torch.float64))
        self.beta = nn.Parameter(torch.zeros(num_inputs, dtype=torch.float64))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs, dtype=torch.float64))
        self.register_buffer('running_var', torch.ones(num_inputs, dtype=torch.float64))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run_date direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device, dtype=torch.float64)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                module.double()
                inputs = inputs.double()
                cond_inputs = cond_inputs.double()
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs=None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_().to(dtype=torch.float64)
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device).to(dtype=torch.float64)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

    # def base_distribution_log_prob(x):
    #     log_probs = (-0.5 * x.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
    #     return log_probs.sum(-1, keepdim=True)
    #
    # def forward_and_log_prob(self, inputs, cond_inputs=None):
    #     u, log_jacob = self(inputs, cond_inputs)
    #     log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
    #     return u, (log_probs + log_jacob).sum(-1, keepdim=True)
    #
    # def fit(self,
    #         data,
    #         context=None,
    #         validation_data=None,
    #         validation_context=None,
    #         validation_split=0.0,
    #         epoch=20,
    #         bs=100,
    #         patience=np.inf,
    #         monitor='val_loss',
    #         shuffle=True,
    #         lr=1e-3,
    #         device='cpu',
    #         verbose=2):
    #     """
    #         Method to fit the normalising flow.
    #
    #     """
    #
    #     optimizer = torch.optim.Adam(self.parameters(), lr)
    #
    #     if not isinstance(data, torch.Tensor):
    #         data = torch.tensor(data, dtype=torch.float32)
    #
    #     if (validation_data is not None) and (not isinstance(validation_data, torch.Tensor)):
    #         validation_data = torch.tensor(validation_data, dtype=torch.float32)
    #
    #     if context is not None:
    #         use_context = True
    #         if not isinstance(data, torch.Tensor):
    #             context = torch.tensor(context, dtype=torch.float32)
    #     else:
    #         use_context = False
    #
    #     if (validation_context is not None) and (not isinstance(validation_context, torch.Tensor)):
    #         validation_context = torch.tensor(validation_context, dtype=torch.float32)
    #
    #     if validation_data is not None:
    #
    #         if use_context:
    #             train_dl = DataLoader(TensorDataset(data, context), bs, shuffle)
    #             val_dl = DataLoader(TensorDataset(validation_data, validation_context), bs, shuffle)
    #         else:
    #             train_dl = DataLoader(TensorDataset(data), bs, shuffle)
    #             val_dl = DataLoader(TensorDataset(validation_data), bs, shuffle)
    #
    #         validation = True
    #     else:
    #         if validation_split > 0.0 and validation_split < 1.0:
    #             validation = True
    #             split = int(data.size()[0] * (1. - validation_split))
    #             if use_context:
    #                 data, validation_data = data[:split], data[split:]
    #                 context, validation_context = context[:split], context[split:]
    #                 train_dl = DataLoader(TensorDataset(data, context), bs, shuffle)
    #                 val_dl = DataLoader(TensorDataset(validation_data, validation_context), bs, shuffle)
    #             else:
    #                 data, validation_data = data[:split], data[split:]
    #                 train_dl = DataLoader(TensorDataset(data), bs, shuffle)
    #                 val_dl = DataLoader(TensorDataset(validation_data), bs, shuffle)
    #         else:
    #             validation = False
    #             if use_context:
    #                 train_dl = DataLoader(TensorDataset(data, context), bs, shuffle)
    #             else:
    #                 train_dl = DataLoader(TensorDataset(data), bs, shuffle)
    #
    #     history = {}  # Collects per-epoch loss
    #     history['loss'] = []
    #     history['val_loss'] = []
    #
    #     if not validation:
    #         monitor = 'loss'
    #     best_epoch = 0
    #     best_loss = np.inf
    #     best_model = copy.deepcopy(self.state_dict())
    #
    #     start_time_sec = time.time()
    #
    #     for epoch in range(epoch):
    #
    #         # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
    #         self.train()
    #         train_loss = 0.0
    #
    #         for batch in train_dl:
    #
    #             optimizer.zero_grad()
    #
    #             if use_context:
    #                 x = batch[0].to(device)
    #                 y = batch[1].to(device)
    #                 loss = -self.log_prob(x, y).mean()
    #             else:
    #                 x = batch[0].to(device)
    #                 loss = -self.log_prob(x).mean()
    #
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(self.parameters(), 1, error_if_nonfinite=True)
    #             optimizer.step()
    #
    #             train_loss += loss.data.item() * x.size(0)
    #
    #         train_loss = train_loss / len(train_dl.dataset)
    #
    #         history['loss'].append(train_loss)
    #
    #         # --- EVALUATE ON VALIDATION SET -------------------------------------
    #         self.eval()
    #         if validation:
    #             val_loss = 0.0
    #
    #             for batch in val_dl:
    #
    #                 if use_context:
    #                     x = batch[0].to(device)
    #                     y = batch[1].to(device)
    #                     loss = -self.log_prob(x, y).mean()
    #                 else:
    #                     x = batch[0].to(device)
    #                     loss = -self.log_prob(x).mean()
    #
    #                 val_loss += loss.data.item() * x.size(0)
    #
    #             val_loss = val_loss / len(val_dl.dataset)
    #
    #             history['val_loss'].append(val_loss)
    #
    #         if verbose > 1:
    #             try:
    #                 print('Epoch %3d/%3d, train loss: %5.2f, val loss: %5.2f' % \
    #                       (epoch + 1, epoch, train_loss, val_loss))
    #             except:
    #                 print('Epoch %3d/%3d, train loss: %5.2f' % \
    #                       (epoch + 1, epoch, train_loss))
    #
    #         # Monitor loss
    #         if history[monitor][-1] < best_loss:
    #             best_loss = history[monitor][-1]
    #             best_epoch = epoch
    #             best_model = copy.deepcopy(self.state_dict())
    #
    #         if epoch - best_epoch >= patience:
    #             self.load_state_dict(best_model)
    #             if verbose > 0:
    #                 print('Finished early after %3d epoch' % (best_epoch))
    #                 print('Best loss achieved %5.2f' % (best_loss))
    #             break
    #
    #     # END OF TRAINING LOOP
    #
    #     if verbose > 0:
    #         end_time_sec = time.time()
    #         total_time_sec = end_time_sec - start_time_sec
    #         time_per_epoch_sec = total_time_sec / epoch
    #         print()
    #         print('Time total:     %5.2f sec' % (total_time_sec))
    #         print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
    #
    #     return history
