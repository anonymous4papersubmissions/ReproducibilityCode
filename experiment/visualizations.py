import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
from tqdm import tqdm

from .train import TrainingExperiment

import strategies
from util import printc

class VisualizationExperiment(TrainingExperiment):
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
        super(VisualizationExperiment, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained, small_init_multiplier, resume, resume_optim, save_freq, contrastive, snapshot_ensemble, beta_lasso)

        resume_dn, _ = os.path.split(resume)
        epoch_counts = []
        resume_pn_li = []
        for resume_fn in os.listdir(resume_dn):
            *resume_prefix, resume_suffix = resume_fn.split('-')
            epoch_counts.append(int(resume_suffix[:-3]))
            resume_pn_li.append(os.path.join(resume_dn, resume_fn))

        self.path = path
        if self.path is not None:
            self.path = pathlib.Path(path)
        printc(f"Logging results to {self.path}", color='MAGENTA')
        self.path.mkdir(exist_ok=True, parents=True)

        self.to_device()

        w_li = []
        num_epochs_diff, = list(set([e-s for e,s in zip(sorted(epoch_counts)[1:],sorted(epoch_counts)[:-1])]))
        for i in sorted(list(range(len(resume_pn_li))), key=lambda x: epoch_counts[x]):
            assert pathlib.Path(resume_pn_li[i]).exists(), "Resume path does not exist"
            previous = torch.load(resume_pn_li[i])
            self.model.load_state_dict(previous['model_state_dict'], strict=True)
            w_li.append(deepcopy(self.model.encoder[0].weight))

        plot_weights([torch.abs(w.view(-1)) for w in w_li], n_bins=32, n_epochs_per_save=num_epochs_diff, path=self.path)

        if dataset in ['SVHN', 'CIFAR10', 'CIFAR100']:
            plot_img(w_li[-1].transpose(0,1), path=self.path)
        elif 'OpenML' in dataset:
            X = self.valid_dataset.data
            y = self.valid_dataset.target
            plot_decision_boundaries(X, y, self.model, n_samples_max=10000, n_samples_grid=1000, path=self.path)
            plot_l1_cossim(w_li, n_epochs_per_save=num_epochs_diff, path=self.path)
            plot_matrix(w_li[-1].transpose(0,1))

        else:
            assert False

def plot_l1_cossim(w_li, n_epochs_per_save=5, path=None):
    '''
    plt.clf()
    x = np.arange(len(w_li))*n_epochs_per_save
    y = []
    for w in w_li:
        grid = torch.eye(w.shape[0], dtype=torch.float, device=w.device) # 5x5
        norm = torch.norm(w, dim=1).view(1,-1) # 512x1
        cossim = torch.abs(torch.matmul(w.transpose(0,1), grid)/norm) # 512x5 * 5x5 / 512x1 = 512x5
        l1_cossim = torch.mean(torch.max(cossim, dim=0)[0], dim=0) # 512
        y.append(l1_cossim.detach().cpu().numpy())
    plt.plot(x, np.array(y))
    plt.xlabel('# Epochs')
    plt.ylabel('Avg. Cossim with unit vectors')
    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, 'l1_cossim.png'))
    '''
    pass

def plot_decision_boundaries(X, y, model, n_samples_max=1000, n_samples_grid=100, batch_size=64, path=None):
    '''
    plt.clf()
    model_surrogate = LogisticRegression()
    X_np, y_np = X.cpu().numpy()[:n_samples_max], y.cpu().numpy()[:n_samples_max]
    model_surrogate.fit(X_np, y_np)

    feat_mean = X_np.mean(axis=0)
    feat_idx1, feat_idx2 = np.argsort(model_surrogate.coef_[0])[-2:]
    # feat_mean1, feat_std1 = X_np[:, feat_idx1].mean(), X_np[:, feat_idx1].std()
    # feat_mean2, feat_std2 = X_np[:, feat_idx2].mean(), X_np[:, feat_idx2].std()
    # feat_span1 = np.linspace(feat_mean1-feat_std1, feat_mean1+feat_std1, n_samples_grid)
    # feat_span2 = np.linspace(feat_mean1-feat_std2, feat_mean1+feat_std2, n_samples_grid)
    feat_min1, feat_max1 = X_np[:, feat_idx1].min()-1, X_np[:, feat_idx1].max()+1
    feat_min2, feat_max2 = X_np[:, feat_idx2].min()-1, X_np[:, feat_idx2].max()+1
    feat_span1 = np.linspace(feat_min1, feat_max1, n_samples_grid)
    feat_span2 = np.linspace(feat_min2, feat_max2, n_samples_grid)
    xx, yy = np.meshgrid(feat_span1, feat_span2)
    model.eval()

    X_in = torch.tensor(feat_mean).unsqueeze(0).repeat(n_samples_grid**2, 1)
    X_in[:,feat_idx1] = torch.from_numpy(xx.ravel())
    X_in[:,feat_idx2] = torch.from_numpy(yy.ravel())

    model = model.to(X_in.device)
    labels_predicted = []
    for idx in tqdm(range(math.ceil(len(X_in)/batch_size))):
        labels_predicted.extend(
            torch.argmax(model(X_in[idx*batch_size:(idx+1)*batch_size]), dim=-1).detach().numpy().tolist())
    # labels_predicted = [0 if value <= 0.0 else 1 for value in labels_predicted]
    z = np.array(labels_predicted).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap='Paired', alpha=0.5)

    # Get predicted labels on training data and plot
    ax.scatter(X[:n_samples_max, 0], X[:n_samples_max, 1], c=y[:n_samples_max].reshape(-1), s=3, cmap='Paired', lw=0)
    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, 'decision_boundaries.png'))
    '''
    pass

def plot_matrix(w, path=None):
    plt.clf()
    MEDIUM_SIZE = 17 
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    plt.imshow(torch.abs(w_slice).detach().cpu().numpy(), cmap='gnuplot', interpolation='nearest', aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Magnitude')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'w_matrix.png'))

def plot_img(w, C=3, H=32, W=32, n_samples=(2,3), path=None):
    plt.clf()
    n_samples_h, n_samples_w = n_samples
    assert n_samples_h*n_samples_w <= w.shape[-1]

    w_reshape = w.view(C, H, W, w.shape[-1])
    for i in range(n_samples_h*n_samples_w):
        plt.axis('off')
        plt.subplot(n_samples_h, n_samples_w, i+1)
        w_slice = torch.abs(w_reshape[:,:,:,i].transpose(0,-1))
        w_slice = (255*w_slice/w_slice.max()).type(torch.int)
        plt.imshow(w_slice.detach().cpu().numpy(), interpolation='nearest', aspect='auto')
        plt.axis('off')
    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, 'img.png'))

def plot_weights(w_li, n_bins=16, n_epochs_per_save=5, path=None):
    plt.clf()
    MEDIUM_SIZE = 17 
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

    w_li = torch.stack(w_li, dim=0)
    w_max = w_li.max().cpu().item()
    w_min = w_li.min().cpu().item()
    out = torch.stack([
        torch.histc(w, bins=n_bins, min=w_min, max=w_max).flip(dims=(0,)) \
        for w in w_li], dim=1).detach().cpu().numpy()
    plt.imshow(out, cmap='Blues', norm=mpl.colors.LogNorm(), interpolation='nearest', aspect='auto', extent=[0,n_epochs_per_save*len(w_li),w_min,w_max])
    # plt.xticks(ticks=plt.xticks()[0][1:], labels=n_epochs_per_save * np.array(plt.xticks()[0][1:], dtype=int))
    plt.xlabel('# Epochs')
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.ylabel('Weight Magnitude')
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, 'weights.png'))

if __name__ == '__main__':
    w_li = [torch.randn(256, 256).view(-1) for _ in range(20)]
    plot_weights(w_li, n_bins=32)

    w = torch.randn(3, 32, 32, 512)
    w = w.view(-1, w.shape[-1])
    plot_img(w)

    model = \
        torch.nn.Sequential(
            torch.nn.Linear(5, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 8), torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )
    X = torch.randn(50000, 5)
    y = (torch.randn(50000) > 0.0).type(torch.long)
    plot_decision_boundaries(X, y, model, n_samples_max=10000, n_samples_grid=1000)

    w_li = [torch.randn(5, 512) for _ in range(20)]
    plot_l1_cossim(w_li)
