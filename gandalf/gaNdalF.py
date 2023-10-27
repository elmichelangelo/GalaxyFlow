from gandalf_galaxie_dataset import DESGalaxies
from torch.utils.data import DataLoader, RandomSampler
from xgboost import XGBClassifier
import torch
import os


class gaNdalF():
    """"""
    def __init__(self, cfg):
        """"""
        self.cfg = cfg

        self.galaxie_loader = self.init_dataset()

        self.gandalf_flow, self.gandalf_classifier = self.init_trained_models()

    def make_dirs(self):
        """"""
        if not os.path.exists(self.cfg['PATH_OUTPUT_RUN']):
            os.mkdir(self.cfg['PATH_OUTPUT_RUN'])
        if self.cfg['PLOT_TEST_RUN'] is True:
            if not os.path.exists(self.cfg['PATH_PLOTS_RUN']):
                os.mkdir(self.cfg['PATH_PLOTS_RUN'])
            for path_plot in self.cfg['PATH_PLOTS_FOLDER_RUN'].values():
                if not os.path.exists(path_plot):
                    os.mkdir(path_plot)

        if self.cfg['SAVE_NN_RUN'] is True:
            if not os.path.exists(self.cfg["PATH_SAVE_NN_RUN"]):
                os.mkdir(self.cfg["PATH_SAVE_NN_RUN"])

    def init_dataset(self):
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="run_gandalf",
            lst_split=[
                self.cfg['SIZE_TRAINING_DATA_RUN'],
                self.cfg['SIZE_VALIDATION_DATA_RUN'],
                self.cfg['SIZE_TEST_DATA_RUN']
            ]
        )
        galaxie_loader = DataLoader(galaxies.tsr_data, shuffle=True, num_workers=0)
        return galaxie_loader

    def init_trained_models(self):
        """"""
        gandalf_flow = torch.load(f"{self.cfg['PATH_Trained_NN']}/{self.cfg['NN_FILE_NAME_FLOW']}")

        gandalf_classifier = XGBClassifier()
        gandalf_classifier.load_model(f"{self.cfg['PATH_Trained_NN']}/{self.cfg['NN_FILE_NAME_CLASSF']}")

        return gandalf_flow, gandalf_classifier

    @staticmethod
    def sample_random_data_from_dataset(dataset, n):
        sampler = RandomSampler(dataset, replacement=True, num_samples=n)
        dataloader = DataLoader(dataset, batch_size=n, sampler=sampler)
        for data in dataloader:
            return data

    def run(self):
        """"""
        samples = self.sample_random_data_from_dataset(dataset=self.galaxie_loader, n=self.cfg['NUMBER_SAMPLES'])
        output_data_gandalf_classifier = self.gandalf_classifier.predict(samples)
        output_data_gandalf_flow = self.gandalf_flow.sample(len(samples), cond_inputs=samples).detach()
