import json
import pathlib
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from .train import TrainingExperiment

import strategies
from experiment.contrastive_loss import SCARF
from experiment.train import NCropTransform
import datasets
from util import printc, OnlineStats

class ConsistencyExperiment(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 strategy,
                 compression,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=True,
                 small_init_multiplier=None,
                 reset_weights=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 contrastive=None,
                 snapshot_ensemble=None,
                 beta_lasso=None):

        super(ConsistencyExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, small_init_multiplier, resume, resume_optim, save_freq, contrastive, snapshot_ensemble, beta_lasso)
        self.add_params(strategy=strategy, compression=compression)

        self.apply_pruning(strategy, compression)

        assert reset_weights
        if reset_weights:
            assert resume is not None
            print('reset weights!')
            *resume_prefix, resume_suffix = resume.split('-')
            resume_init = '-'.join(resume_prefix) + '-0.pt'
            print(resume_init)
            assert pathlib.Path(resume_init).exists(), "Resume path does not exist"
            previous = torch.load(resume_init)
            self.model.load_state_dict(previous['model_state_dict'], strict=False)

        self.cloader_dict = self.build_cloader_wrapper(dataset, contrastive, dl_kwargs)

        self.path = path
        if self.path is not None:
            self.path = pathlib.Path(path)
        self.save_freq = save_freq
        self.run()

    def apply_pruning(self, strategy, compression):
        constructor = getattr(strategies, strategy)
        x, y = next(iter(self.train_dl))
        self.pruning = constructor(self.model, x, y, compression=compression)
        if compression > 1:
            self.pruning.apply() # model is masked here
        else:
            print('skip pruning!')
        printc("Masked model", color='GREEN')

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.build_logging([f'{nm}_consistency' for nm in self.cloader_dict.keys()], self.path)
        self.to_device()
        self.run_consistency_wrapper()
        self.csvlogger.update()

    def build_cloader_wrapper(self, dataset, contrastive, dl_kwargs):
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}

        assert contrastive is not None
        n_views, transform_cfg = contrastive.split(',')
        assert int(n_views) == 2

        self.contrastive = int(n_views), transform_cfg
        if 'batchconstant' in transform_cfg:
            print('batch size stays the same')
            pass
        else:
            dl_kwargs['batch_size'] = int(dl_kwargs['batch_size']/int(n_views))
        self.scarf = SCARF(int(n_views))
        return self.build_cloader(dataset, **dl_kwargs)

    def build_cloader(self, dataset, **dl_kwargs):
        assert self.contrastive is not None
        n_views, transform_cfg = self.contrastive
        cloader_dict = {}
        if 'simclr' in transform_cfg:
            simclr_transform_li, simclr_tranform_nm_li = self.get_simclr_transforms()
            for transform, nm in zip(simclr_transform_li, simclr_tranform_nm_li):
                transform = \
                    transforms.Compose([transform, transforms.ToTensor()])
                transform = NCropTransform(transform, n_views)
                constructor = getattr(datasets, dataset)
                valid_dataset = constructor(tvt='valid', transform=transform)
                valid_dl = DataLoader(valid_dataset, shuffle=False, **dl_kwargs)
                cloader_dict[nm] = valid_dl
        elif 'LoCL' in transform_cfg:
            constructor = getattr(datasets, dataset)
            valid_dataset = constructor(tvt='valid', transform=None)
            valid_dataset.n_views_contrastive = int(self.contrastive[0])
            cloader_dict['scarf'] = DataLoader(valid_dataset, shuffle=False, **dl_kwargs)
        else:
            assert False
        return cloader_dict

    def run_consistency_wrapper(self, epoch=0):
        for prefix, cloader in self.cloader_dict.items():
            cons = OnlineStats()
            epoch_iter = tqdm(cloader)
            epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")
            for i, (x, y) in enumerate(epoch_iter, start=1):
                consistency, yhat, _ = self.run_consistency(x)
                cons.add(consistency)
            self.log(**{
                f'{prefix}_consistency': cons.mean,
            })

    def run_consistency(self, x):
        # Contrastive Train Loop
        _, contrastive_cfg = self.contrastive

        if 'simclr' in contrastive_cfg:
            yhat_li = []
            g1_x, g2_x = x
        elif 'LoCL' in contrastive_cfg:
            x = x.to(self.device)
            assert len(x.shape) == 3
            tab_anchor, tab_random = x[:, 0, :], x[:, 1:, :]
            g1_x, (g2_x, *_) = self.scarf(tab_anchor, tab_random.transpose(0, 1))
        else:
            assert False
        g1_x = g1_x.to(self.device)
        g2_x = g2_x.to(self.device)
        g1_yhat, g2_yhat = self.model(g1_x).argmax(dim=-1), self.model(g2_x).argmax(dim=-1)
        consistency = torch.eq(g1_yhat, g2_yhat).type(torch.float).mean()
        return consistency, g1_yhat, g2_yhat
