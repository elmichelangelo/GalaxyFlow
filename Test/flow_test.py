import unittest

import torch

import galaxyflow.flow as fnn

EPS = 1e-5
BATCH_SIZE = 32
NUM_INPUTS = 11
NUM_HIDDEN = 64
mask = torch.arange(0, NUM_INPUTS) % 2
mask = mask.unsqueeze(0)


class TestFlow(unittest.TestCase):
    def testBatchNorm(self):
        m1 = fnn.FlowSequential(fnn.BatchNormFlow(NUM_INPUTS))
        m1.train()

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'BatchNorm Det is not zero.')
        self.assertTrue((x - z).abs().max() < EPS, 'BatchNorm is wrong.')

        # Second run.
        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'BatchNorm Det is not zero for the second run.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'BatchNorm is wrong for the second run.')

        m1.eval()
        m1 = fnn.FlowSequential(fnn.BatchNormFlow(NUM_INPUTS))

        x = torch.randn(BATCH_SIZE, NUM_INPUTS)

        y, logdets = m1(x)
        z, inv_logdets = m1(y, mode='inverse')

        self.assertTrue((logdets + inv_logdets).abs().max() < EPS,
                        'BatchNorm Det is not zero in eval.')
        self.assertTrue((x - z).abs().max() < EPS,
                        'BatchNorm is wrong in eval.')


if __name__ == "__main__":
    unittest.main()