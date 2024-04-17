import json
import pathlib
import torch

from .train import TrainingExperiment

import strategies
from metrics import model_size, flops
from util import printc


class PruningExperiment(TrainingExperiment):

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

        super(PruningExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, small_init_multiplier, resume, resume_optim, save_freq, contrastive, snapshot_ensemble, beta_lasso)
        self.add_params(strategy=strategy, compression=compression)

        self.apply_pruning(strategy, compression)

        if reset_weights:
            assert resume is not None
            print('reset weights!')
            *resume_prefix, resume_suffix = resume.split('-')
            resume_init = '-'.join(resume_prefix) + '-0.pt'
            print(resume_init)
            assert pathlib.Path(resume_init).exists(), "Resume path does not exist"
            previous = torch.load(resume_init)
            self.model.load_state_dict(previous['model_state_dict'], strict=False)

        self.path = path
        if self.path is not None:
            self.path = pathlib.Path(path)
        self.save_freq = save_freq

    def apply_pruning(self, strategy, compression):
        constructor = getattr(strategies, strategy)
        x, y = next(iter(self.train_dl))
        self.pruning = constructor(self.model, x, y, compression=compression)
        if compression > 1:
            self.pruning.apply() # model is masked here
            # for layer in self.model.children():
            #     if hasattr(layer, 'reset_parameters'):
            #         layer.reset_parameters()
        else:
            print('skip pruning!')
        printc("Masked model", color='GREEN')

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)

        self.save_metrics()

        # if self.pruning.compression > 1:
        self.run_epochs()

    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        self.metrics['flops'] = int(self.metrics['flops'])
        # print({k:(v,type(v)) for k,v in self.metrics.items()})
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        printc(json.dumps(self.metrics, indent=4), color='GRASS')
        summary = self.pruning.summary()
        summary_path = self.path / 'masks_summary.csv'
        summary.to_csv(summary_path)
        print(summary)

    def pruning_metrics(self):

        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.valid_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        loss, acc1_valid, acc5_valid = self.run_epoch('valid', -1)
        _, acc1_test, acc5_test = self.run_epoch('test', -1)
        self.log_epoch(-1)

        metrics['loss'] = loss
        metrics['valid_acc1'] = acc1_valid
        metrics['valid_acc5'] = acc5_valid
        metrics['test_acc1'] = acc1_test
        metrics['test_acc5'] = acc5_test

        return metrics
