import pathlib
import time

import os
import copy
import math

import numpy as np
import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import autocast
import json

from sklearn.metrics import balanced_accuracy_score
from experiment.base import Experiment
from experiment.contrastive_loss import NTXent, SCARF
from experiment.snapshot_ensembles import CyclicLRWithRestarts
import datasets
import models
from metrics import correct
from models.head import mark_classifier
from util import printc, OnlineStats

def permute(a):
    f = lambda x: x.detach().cpu().numpy()
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    from scipy import sparse
    sa = sparse.csr_matrix(f(a))
    o = reverse_cuthill_mckee(sa)
    out = f(a[o.tolist(),:][:,o.tolist()])
    return out, o

class NCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def append_normalize(self, normalize):
        self.transform = transforms.Compose([self.transform, normalize])

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n)]

class TrainingExperiment(Experiment):

    default_dl_kwargs = {'batch_size': 128,
                         'pin_memory': False,
                         'num_workers': 2
                         }

    default_train_kwargs = {'optim': 'SGD',
                            'epochs': 30,
                            'lr': 1e-3,
                            }

    def __init__(self,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=False,
                 small_init_multiplier=None,
                 resume=None,
                 resume_optim=False,
                 save_freq=10,
                 contrastive=None,
                 snapshot_ensemble=None,
                 beta_lasso=None):

        # Default children kwargs
        super(TrainingExperiment, self).__init__(seed)
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        train_kwargs = {**self.default_train_kwargs, **train_kwargs}

        self.snapshot_ensemble = snapshot_ensemble
        self.model_li = []

        self.beta_lasso = beta_lasso

        if contrastive is None:
            self.contrastive = None
        else:
            n_views, transform_cfg = contrastive.split(',')
            self.contrastive = int(n_views), transform_cfg
            self.ntxent = NTXent()
            if 'batchconstant' in transform_cfg:
                print('batch size stays the same')
                pass
            else:
                dl_kwargs['batch_size'] = int(dl_kwargs['batch_size']/int(n_views))
            self.scarf = SCARF(int(n_views))

        params = locals()
        params['dl_kwargs'] = dl_kwargs
        params['train_kwargs'] = train_kwargs

        self.add_params(**params)
        self.model_name = model
        # Save params

        self.resume = resume
        self.build_dataloader(dataset, **dl_kwargs)

        num_features = None
        if dataset == 'SVHN':
            num_classes = 10
        elif dataset == 'CIFAR10':
            num_classes = 10
        elif dataset == 'CIFAR100':
            num_classes = 100
        elif 'OpenML' in dataset:
            num_features = self.train_dataset.num_features
            num_classes = self.train_dataset.num_classes
        else:
            assert False
        self.build_model(
            model=model, num_classes=num_classes, pretrained=pretrained,
            small_init_multiplier=small_init_multiplier, resume=resume, num_features=num_features)
        print(f'num classes: {num_classes}')
        self.build_train(resume_optim=resume_optim, batch_size=dl_kwargs['batch_size'], **train_kwargs)

        self.path = path
        self.save_freq = save_freq

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)
        self.run_epochs()

    def build_dataloader(self, dataset, **dl_kwargs):
        if self.contrastive is not None:
            n_views, transform_cfg = self.contrastive
            transform_train = self.get_contrastive_views(transform_cfg)
            if transform_train is None:
                transform = None
            else:
                transform = NCropTransform(transform_train, n_views)
        else:
            transform = None
        constructor = getattr(datasets, dataset)

        self.train_dataset = constructor(tvt='train', transform=transform)
        self.valid_dataset = constructor(tvt='valid')
        self.test_dataset = constructor(tvt='test')
        if self.contrastive is not None and 'LoCL' in self.contrastive[1]:
            self.train_dataset.n_views_contrastive = int(self.contrastive[0])

        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)
        self.valid_dl = DataLoader(self.valid_dataset, shuffle=False, **dl_kwargs)
        self.test_dl = DataLoader(self.test_dataset, shuffle=False, **dl_kwargs)

    def get_simclr_transforms(self):
        simclr_transform_li = \
            [
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        simclr_tranform_nm_li = ['resize_crop', 'horizontal_flip', 'color_jitter', 'random_grayscale']
        assert len(simclr_transform_li) == len(simclr_tranform_nm_li)
        return simclr_transform_li, simclr_tranform_nm_li

    def get_contrastive_views(self, transform_cfg='simclr'):
        if 'simclr' in transform_cfg:
            transform_train = transforms.Compose(
                self.get_simclr_transforms()[0] + [transforms.ToTensor()]
                # transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2047, 0.2435, 0.2616)),
            )
        elif 'LoCL' in transform_cfg:
            transform_train = None
        else:
            assert False

        return transform_train

    def build_model(self, model, num_classes, pretrained=True, small_init_multiplier=None, resume=None, num_features=None):
        if isinstance(model, str):
            if hasattr(models, model):
                if 'mlpnet' in model:
                    assert num_features is not None, 'we only have this parameter for tabular datasets'
                    model = getattr(models, model)(num_features=num_features, num_classes=num_classes)
                else:
                    model = getattr(models, model)(pretrained=pretrained, num_classes=num_classes)

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)(pretrained=pretrained, num_classes=num_classes)
                mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")

        self.model = model

        if resume is not None:
            self.resume = pathlib.Path(self.resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous['model_state_dict'])

        if small_init_multiplier is not None:
            small_init_multiplier = float(small_init_multiplier)
            assert small_init_multiplier >= 0.0
            print(f'applying small_init_multiplier={small_init_multiplier}')
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.data = small_init_multiplier * param

    def build_train(self, optim, epochs, resume_optim=False, batch_size=None, **optim_kwargs):
        default_optim_kwargs = {
            'SGD': {'momentum': 0.9, 'nesterov': True, 'lr': 1e-3},
            'Adam': {'momentum': 0.9, 'betas': (.9, .99), 'lr': 1e-4}
        }

        self.epochs = epochs

        # Optim
        if self.snapshot_ensemble is not None:
            optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
            self.scheduler = CyclicLRWithRestarts(
                optim, batch_size, math.ceil(len(self.train_dataset)/batch_size),
                restart_period=15, t_mult=2, policy="cosine")
            self.restart_count = None
        else:
            if isinstance(optim, str):
                constructor = getattr(torch.optim, optim)
                if optim in default_optim_kwargs:
                    optim_kwargs = {**default_optim_kwargs[optim], **optim_kwargs}
                optim = constructor(self.model.parameters(), **optim_kwargs)
            self.scheduler = None

        self.optim = optim

        if resume_optim:
            assert hasattr(self, "resume"), "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous['optim_state_dict'])

        # Assume classification experiment
        self.loss_func = nn.CrossEntropyLoss()

    def to_device(self):
        # Torch CUDA config
        self.device = torch.device((
               f'cuda:{os.environ["CUDA_VISIBLE_DEVICE"]}' if \
               "CUDA_VISIBLE_DEVICE" in os.environ else 'cuda'
           ) if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="ORANGE")
        self.model.to(self.device)
        cudnn.benchmark = True   # For fast training.

    def checkpoint(self):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, checkpoint_path / f'checkpoint-{epoch}.pt')

    def run_epochs(self):
        since = time.time()
        try:
            for epoch in range(self.epochs):
                printc(f"Start epoch {epoch}", color='YELLOW')
                self.train(epoch)
                self.valid(epoch)
                self.test(epoch)
                if epoch % self.save_freq == 0:
                    self.checkpoint()
                self.log(timestamp=time.time()-since)
                self.log_epoch(epoch)

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def get_features(self, img):
        if 'dfcnet' in self.model_name:
            z = self.model.features(img)
        elif 'resnet' in self.model_name:
            x = self.model.conv1(img)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            z = self.model.avgpool(x).reshape(x.shape[0], -1)
            # z = self.model.features(img)
        elif 'vgg' in self.model_name:
            z = self.model.features(img)
        elif 'mlpnet' in self.model_name:
            z = self.model.features(img)
        elif 'cnn' in self.model_name:
            z = self.model.features(img)
        else:
            assert False
        return z

    def get_loss_scl(self, x, y):
        # Contrastive Train Loop
        _, contrastive_cfg = self.contrastive

        if 'simclr' in contrastive_cfg:
            y = y.to(self.device)
            z_li = []
            for x_in in x:
                z = self.get_features(x_in.to(self.device))
                z_li.append(z)
            z = torch.stack(z_li, dim=1)

            x_in = x[0].to(self.device)
            loss_scl = self.ntxent(z, y)
        elif 'LoCL' in contrastive_cfg:
            x = x.to(self.device)
            y = y.to(self.device)

            assert len(x.shape) == 3
            tab_anchor, tab_random = x[:, 0, :], x[:, 1:, :]
            tab_anchor, tab_view_li = self.scarf(tab_anchor, tab_random.transpose(0, 1))
            z_li = []
            for tab in torch.cat([tab_anchor.unsqueeze(0), tab_view_li], dim=0):
                tab = tab.to(self.device)
                z = self.get_features(tab)
                z_li.append(z)
            z = torch.stack(z_li, dim=1)

            x_in = tab_anchor.to(self.device)
            loss_scl = self.ntxent(z, y)
        else:
            assert False
        return loss_scl, x_in

    def run_epoch(self, tvt, epoch=0):
        if self.scheduler is not None:
            self.scheduler.step()
        save_snapeshot = None
        if self.snapshot_ensemble:
            if self.restart_count is None:
                save_snapshot = False
            else:
                save_snapeshot = self.restart_count != self.scheduler.restarts
            self.restart_count = self.scheduler.restarts

        if tvt == 'train':
            train = True
            prefix = 'train'
            dl = self.train_dl
        elif tvt == 'valid':
            train = False
            prefix = 'valid'
            dl = self.valid_dl
        elif tvt == 'test':
            train = False
            prefix = 'test'
            dl = self.test_dl
        else:
            assert False, f'tvt: {tvt}'

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")

        if train:
            self.model.train()
        else:
            self.model.eval()

        ypred_li = []
        ytrue_li = []
        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):
                # Contrastive Train Loop
                if train and self.contrastive is not None:
                    loss_scl, x_in = self.get_loss_scl(x, y)
                else:
                    loss_scl = None
                    x_in = x

                # Beta Lasso Train Loop
                if self.beta_lasso is not None:
                    lmbda = float(self.beta_lasso)
                    l1_reg = torch.tensor(0.0)
                    for w in self.model.parameters():
                        l1_reg = l1_reg + torch.abs(w).sum()
                    loss_beta_lasso = lmbda * l1_reg
                else:
                    loss_beta_lasso = None

                # Normal Train Loop
                x_in, y = x_in.to(self.device), y.to(self.device)
                if not train and self.snapshot_ensemble is not None:
                    yhat_li = [model(x_in).unsqueeze(0) for model in self.model_li + [self.model]]
                    yhat = torch.mean(torch.cat(yhat_li), 0).squeeze()
                else:
                    yhat = self.model(x_in)

                if len(yhat.shape) == 1:
                    yhat = yhat.unsqueeze(0)
                loss_sup = self.loss_func(yhat, y)

                if train and self.beta_lasso is not None:
                    assert loss_beta_lasso is not None
                    loss = loss_sup + loss_beta_lasso
                elif train and self.contrastive is not None:
                    assert loss_scl is not None
                    _, contrastive_cfg = self.contrastive
                    if '+' in contrastive_cfg:
                        loss = loss_scl + loss_sup
                    else:
                        loss = loss_scl
                else:
                    loss = loss_sup

                if train:
                    loss.backward()

                    self.optim.step()
                    self.optim.zero_grad()

                    if self.beta_lasso is not None:
                        beta = 50*float(self.beta_lasso)
                        # prune weights larger than some number...
                        with torch.no_grad():
                            for w in self.model.parameters():
                                w *= (torch.abs(w)>beta).type(torch.float)

                total_loss.add(loss.item() / dl.batch_size)
                if yhat is not None:
                    c1, c5 = correct(yhat, y, (1, min(5, yhat.shape[1])))
                    acc1.add(c1 / yhat.shape[0])
                    acc5.add(c5 / yhat.shape[0])
                    _, ypred = yhat.topk(k=1, dim=1, largest=True, sorted=True)
                    ypred_li.append(ypred.flatten().detach().cpu().numpy())
                    ytrue_li.append(y.detach().cpu().numpy())

                epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        if len(ypred_li) > 0:
            acc_bal = balanced_accuracy_score(np.concatenate(ypred_li), np.concatenate(ytrue_li))
        else:
            acc_bal = -1

        self.log(**{
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
            f'{prefix}_acc_bal': acc_bal,
        })

        if self.snapshot_ensemble and save_snapeshot:
            # print(f'snapshotting model at epoch {epoch}')
            model_snapshot = copy.deepcopy(self.model)
            model_snapshot.eval()
            self.model_li.append(model_snapshot)

        return total_loss.mean, acc1.mean, acc5.mean

    def train(self, epoch=0):
        return self.run_epoch('train', epoch)

    def valid(self, epoch=0):
        return self.run_epoch('valid', epoch)

    def test(self, epoch=0):
        return self.run_epoch('test', epoch)

    @property
    def train_metrics(self):
        return ['epoch', 'timestamp',
                'train_loss', 'train_acc1', 'train_acc5', 'train_acc_bal',
                'valid_loss', 'valid_acc1', 'valid_acc5', 'valid_acc_bal',
                'test_loss', 'test_acc1', 'test_acc5', 'test_acc_bal',
                ]

    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], torch.nn.Module):
            self.params['model'] = self.params['model'].__module__
        
        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)
