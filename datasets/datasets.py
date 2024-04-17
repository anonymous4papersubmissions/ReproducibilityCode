import pathlib
import os

import numpy as np
import openml
import torch
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, Dataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

from . import places365

_constructors = {
    'MNIST': datasets.MNIST,
    'CIFAR10': datasets.CIFAR10,
    'CIFAR100': datasets.CIFAR100,
    'ImageNet': datasets.ImageNet,
    'Places365': places365.Places365,
    'SVHN': datasets.SVHN,
}


def dataset_path(dataset, path=None):
    """Get the path to a specified dataset

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        dataset_path -- pathlib.Path for the first match

    Raises:
        ValueError -- If no path is provided and DATAPATH is not set
        LookupError -- If the given dataset cannot be found
    """
    if path is None:
        # Look for the dataset in known paths
        if 'DATAPATH' in os.environ:
            path = os.environ['DATAPATH']
            paths = [pathlib.Path(p) for p in path.split(':')]
        else:
            raise ValueError(f"No path specified. A path must be provided, \n \
                           or the folder must be listed in your DATAPATH")

    paths = [pathlib.Path(p) for p in path.split(':')]

    for p in paths:
        p = (p / dataset).resolve()
        if p.exists():
            print(f"Found {dataset} under {p}")
            return p
    else:
        raise LookupError(f"Could not find {dataset} in {paths}")


def dataset_builder(dataset, tvt='train', normalize=None, preproc=None, path=None, transform=None):
    """Build a torch.utils.Dataset with proper preprocessing

    Arguments:
        dataset {str} -- One of MNIST, CIFAR10, CIFAR100, ImageNet, Places365

    Keyword Arguments:
        train {bool} -- Whether to return train or validation set (default: {True})
        normalize {torchvision.Transform} -- Transform to normalize data channel wise (default: {None})
        preproc {list(torchvision.Transform)} -- List of preprocessing operations (default: {None})
        path {str} -- Semicolon separated list of paths to look for dataset folders (default: {None})

    Returns:
        torch.utils.data.Dataset -- Dataset object with transforms and normalization
    """
    if transform is None:
        if preproc is not None:
            preproc += [transforms.ToTensor()]
            if normalize is not None:
                preproc += [normalize]
            preproc = transforms.Compose(preproc)
    else:
        transform.append_normalize(normalize)
        preproc = transform

    path = dataset_path(dataset, path)

    kwargs = {'transform': preproc}
    if dataset == 'ImageNet':
        if tvt == 'train':
            kwargs['split'] = 'train'
        elif tvt == 'valid':
            kwargs['split'] = 'val'
        elif tvt == 'test':
            kwargs['split'] = 'test'
        else:
            assert False
        dataset = _constructors[dataset](path, download=True, **kwargs)
    else:
        if tvt == 'train' or tvt == 'valid':
            if dataset == 'SVHN':
                kwargs['split'] = 'train'
            else:
                kwargs['train'] = True
            dataset = _constructors[dataset](path, download=True, **kwargs)
            valid_len = int(len(dataset) * 0.2)
            dataset_train, dataset_valid = torch.utils.data.random_split(
                dataset, [len(dataset)-valid_len, valid_len],
                generator=torch.Generator().manual_seed(1))
            if tvt == 'train':
                dataset = dataset_train
            elif tvt == 'valid':
                dataset = dataset_valid
            else:
                assert False
        elif tvt == 'test':
            if dataset == 'SVHN':
                kwargs['split'] = 'test'
            else:
                kwargs['train'] = False
            dataset = _constructors[dataset](path, download=True, **kwargs)
        else:
            assert False

    return dataset

def MNIST(tvt='train', path=None, **kwargs):
    """Thin wrapper around torchvision.datasets.CIFAR10
    """
    mean, std = 0.1307, 0.3081
    normalize = transforms.Normalize(mean=(mean,), std=(std,))
    dataset = dataset_builder('MNIST', tvt, normalize, [], path, **kwargs)
    dataset.shape = (1, 28, 28)
    return dataset


def CIFAR10(tvt='train', path=None, **kwargs):
    """Thin wrapper around torchvision.datasets.CIFAR10
    """
    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
    normalize = transforms.Normalize(mean=mean, std=std)
    if tvt == 'train':
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder('CIFAR10', tvt, normalize, preproc, path, **kwargs)
    dataset.shape = (3, 32, 32)
    return dataset


def SVHN(tvt='train', path=None, **kwargs):
    """Thin wrapper around torchvision.datasets.CIFAR10
    """
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if tvt == 'train':
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder('SVHN', tvt, normalize, preproc, path, **kwargs)
    dataset.shape = (3, 32, 32)
    return dataset

def CIFAR100(tvt='train', path=None, **kwargs):
    """Thin wrapper around torchvision.datasets.CIFAR100
    """
    mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]
    normalize = transforms.Normalize(mean=mean, std=std)
    if tvt == 'train':
        preproc = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4)]
    else:
        preproc = []
    dataset = dataset_builder('CIFAR100', tvt, normalize, preproc, path, **kwargs)
    dataset.shape = (3, 32, 32)
    return dataset


def ImageNet(tvt='train', path=None, **kwargs):
    """Thin wrapper around torchvision.datasets.ImageNet
    """
    # ImageNet loading from files can produce benign EXIF errors
    import warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if tvt == 'train':
        preproc = [transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder('ImageNet', tvt, normalize, preproc, path, **kwargs)
    dataset.shape = (3, 224, 224)
    return dataset


def Places365(tvt='train', path=None, **kwargs):
    """Thin wrapper around .datasets.places365.Places365
    """

    # Note : Bolei used the normalization for Imagenet, not the one for Places!
    # # https://github.com/CSAILVision/places365/blob/master/train_placesCNN.py
    # So these are kept so weights are compatible
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    normalize = transforms.Normalize((mean,), (std,))
    if tvt == 'train':
        preproc = [transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip()]
    else:
        preproc = [transforms.Resize(256), transforms.CenterCrop(224)]
    dataset = dataset_builder('Places365', tvt, normalize, preproc, path, **kwargs)
    dataset.shape = (3, 224, 224)
    return dataset

import pathlib
import os

import openml
import torch
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def OpenML233090(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233090, tvt=tvt)
def OpenML233091(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233091, tvt=tvt)
def OpenML233092(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233092, tvt=tvt)
def OpenML233093(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233093, tvt=tvt)
def OpenML233088(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233088, tvt=tvt)
def OpenML233094(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233094, tvt=tvt)
def OpenML233096(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233096, tvt=tvt)
def OpenML233099(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233099, tvt=tvt)
def OpenML233102(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233102, tvt=tvt)
def OpenML233103(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233103, tvt=tvt)
def OpenML233104(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233104, tvt=tvt)
def OpenML233106(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233106, tvt=tvt)
def OpenML233107(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233107, tvt=tvt)
def OpenML233108(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233108, tvt=tvt)
def OpenML233109(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233109, tvt=tvt)
def OpenML233110(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233110, tvt=tvt)
def OpenML233112(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233112, tvt=tvt)
def OpenML233113(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233113, tvt=tvt)
def OpenML233114(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233114, tvt=tvt)
def OpenML233115(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233115, tvt=tvt)
def OpenML233116(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233116, tvt=tvt)
def OpenML233117(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233117, tvt=tvt)
def OpenML233118(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233118, tvt=tvt)
def OpenML233119(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233119, tvt=tvt)
def OpenML233120(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233120, tvt=tvt)
def OpenML233121(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233121, tvt=tvt)
def OpenML233122(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233122, tvt=tvt)
def OpenML233123(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233123, tvt=tvt)
def OpenML233124(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233124, tvt=tvt)
def OpenML233126(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233126, tvt=tvt)
def OpenML233130(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233130, tvt=tvt)
def OpenML233131(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233131, tvt=tvt)
def OpenML233132(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233132, tvt=tvt)
def OpenML233133(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233133, tvt=tvt)
def OpenML233134(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233134, tvt=tvt)
def OpenML233135(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233135, tvt=tvt)
def OpenML233137(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233137, tvt=tvt)
def OpenML233142(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233142, tvt=tvt)
def OpenML233143(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233143, tvt=tvt)
def OpenML233146(tvt='train', path=None, **kwargs):
    return _get_openml_wrapper(233146, tvt=tvt)

def _get_openml_wrapper(task_id, tvt='train', apply_one_hot_encoding=False, val_size=0.2, test_size=0.2, **kwargs):
    assert task_id in [233090, 233091, 233092, 233093, 233088, 233094, 233096, 233099, 233102, 233103, 233104, 233106, 233107, 233108, 233109, 233110, 233112, 233113, 233114, 233115, 233116, 233117, 233118, 233119, 233120, 233121, 233122, 233123, 233124, 233126, 233130, 233131, 233132, 233133, 233134, 233135, 233137, 233142, 233143, 233146]
    # credit: https://github.com/releaunifreiburg/WellTunedSimpleNets/blob/be5b3f3ccd3a5000b5108c19dc65c05357428fbc/utilities.py
    task = openml.tasks.get_task(task_id=task_id)
    dataset = task.get_dataset()
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )


    label_encoder = LabelEncoder()
    empty_features = []

    # remove nan features from the dataframe
    nan_columns = X.isna().all()
    for col_index, col_status in enumerate(nan_columns):
        if col_status:
            empty_features.append(col_index)
    # if there are null categorical columns, remove them
    # from the categorical column indicator.
    if len(empty_features) > 0:
        for feature_index in sorted(empty_features, reverse=True):
            del categorical_indicator[feature_index]

    column_names = list(X.columns)
    # delete empty feature columns.
    # Normally this would be done by the simple imputer, but
    # since now it is conditional, we do it ourselves.
    empty_feature_names = [column_names[feat_index] for feat_index in empty_features]
    if any(nan_columns):
        X.drop(labels=empty_feature_names, axis='columns', inplace=True)

    column_names = list(X.columns)
    numerical_columns = []
    categorical_columns = []

    index = 0
    categorical_col_indices = []
    for cat_column_indicator, column_name in zip(categorical_indicator, column_names):
        if cat_column_indicator:
            categorical_columns.append(column_name)
            categorical_col_indices.append(index)
        else:
            numerical_columns.append(column_name)
        index += 1

    transformers = []

    if len(numerical_columns) > 0:
        numeric_transformer = Pipeline(
            steps=[
                ('num_imputer', SimpleImputer(strategy='constant')),
                ('scaler', StandardScaler())
            ]
        )
        transformers.append(('num', numeric_transformer, numerical_columns))

    if not apply_one_hot_encoding:
        for column in categorical_columns:
            X.loc[:, column] = pd.factorize(X.loc[:, column])[0] + 1
    if len(categorical_columns) > 0:
        steps=[
            ('cat_imputer', SimpleImputer(strategy='constant')),
        ]
        if apply_one_hot_encoding:
            steps.append(('cat_encoding', OneHotEncoder(handle_unknown='ignore')))
        else:
            pass
            # steps.append(('cat_encoding', LabelEncoder()))
        categorical_transformer = Pipeline(
            steps=steps,
        )
        transformers.append(('cat', categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
    )

    # label encode the targets
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y, test_size=test_size, random_state=1, stratify=y, shuffle=True)
    X_train, X_val, y_train, y_val = \
        train_test_split(
            X_train, y_train, test_size=val_size, random_state=1, stratify=y_train, shuffle=True)

    preprocessor.fit(X_train, y_train)
    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    train_set = \
        TensorDatasetCustom(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
    valid_set = \
        TensorDatasetCustom(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
    test_set = \
        TensorDatasetCustom(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )

    if tvt == 'train':
        dataset = train_set
    elif tvt == 'valid':
        dataset = valid_set
    elif tvt == 'test':
        dataset = test_set
    else:
        assert False
    dataset.num_features = len(X.columns)
    dataset.num_classes = y.max()+1
    return dataset


# from https://github.com/clabrugere/pytorch-scarf
class TensorDatasetCustom(Dataset):
    def __init__(self, data, target, columns=None):
        self.data = data
        self.target = target
        self.columns = columns
        self.n_views_contrastive = None

    def getitem_contrastive(self, index):
        random_sample_li = []
        for _ in range(self.n_views_contrastive-1):
            random_idx = np.random.randint(0, len(self))
            random_sample = self.data[random_idx] # torch.tensor(self.data[random_idx], dtype=torch.float)
            random_sample_li.append(random_sample)
        sample = self.data[index] # torch.tensor(self.data[index], dtype=torch.float)
        label = self.target[index]

        return torch.stack([sample]+random_sample_li, dim=0), label

    def getitem(self, index):
        sample =self.data[index] #, dtype=torch.float)
        label = self.target[index]
        return sample, label

    def __getitem__(self, index):
        if self.n_views_contrastive is None:
            return self.getitem(index)
        else:
            return self.getitem_contrastive(index)

    def __len__(self):
        return len(self.data)
