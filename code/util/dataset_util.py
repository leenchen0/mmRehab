from os.path import join as fullfile

from torch.utils.data import ConcatDataset

from util.datasets.mmrehab import MmRehab
from util.logger import log


def get_one_dataset(dataset, train=True, val=False):
    fmt = dataset.format
    fmt_cfg = dataset[fmt]
    data_path = [(fullfile(dataset.dir, dataset_name) if isinstance(dataset_name, str) else {
        'path': fullfile(dataset.dir, dataset_name['path']),
        'poses': dataset_name['poses']
    }) for dataset_name in (dataset.training_sets if train or val else dataset.test_sets)]
    if fmt == 'mmrehab':
        return MmRehab(data_path, train, val, **fmt_cfg)
    raise Exception(f'Invalid Dataset Format: {fmt}')


def get_datasets(datasets, train=True, val=False):
    return ConcatDataset([get_one_dataset(dataset, train, val) for dataset in datasets if (len(dataset.training_sets if train or val else dataset.test_sets) > 0)])


def get_dataset(config, train=True, val=False):
    if 'dataset' in config:
        return get_one_dataset(config.dataset, train, val)
    if 'datasets' in config:
        return get_datasets(config.datasets, train, val)

