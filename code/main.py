import os
from os.path import join as fullfile
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from util import common_utils
from util.logger import Logger, log, init as initLogger
from util.dataset_util import get_dataset
from models import kd_model
from trainer.mmrehab_trainer import MmrehabTrainer


def main(args):
    log('Args: ', args)
    config = common_utils.load_config(args.root_path)
    config.args = args
    log("Config: ", config)
    common_utils.set_seed(config.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device

    log('Device: ', device)

    output_dir = fullfile(args.root_path, 'outputs')
    config.output_dir = output_dir

    log("Creating model")
    features = 0

    if 'dataset' in config:
        out_channels = config.dataset[config.dataset.format].output_dim
        features = config.dataset[config.dataset.format].feat_dim
    else:
        out_channels = config.datasets[0][config.datasets[0].format].output_dim
        
    model = kd_model.Model(config, in_channels=features, out_channels=out_channels)
    
    trainer = MmrehabTrainer(config, model)

    model_path = fullfile(args.root_path if "path" not in config.model else config.model.path, f'saved_models/model{args.suffix}')
    common_utils.make_folder(os.path.dirname(model_path))

    if args.finetune:
        model.load_state_dict(torch.load(model_path + '-finetune.pt', 'cpu'))
    model.to(device)

    log("Creating data loaders")
    batch_size = config.training.batch_size
    dataset_test = get_dataset(config, train=False)
    data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.training:
        dataset_train = get_dataset(config, train=True)
        dataset_eval = get_dataset(config, train=False, val=True)
        data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        data_loader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True if len(dataset_eval) > 64 * 64 else False, num_workers=args.workers, pin_memory=True)

        log("Start training")
        model = trainer.train(model_path, data_loader_train, data_loader_eval, data_loader_test, args.recover)

    print("Start testing")
    if 'dataset' in config:
        test_scene = config.dataset[config.dataset.format].test_scene
    else:
        test_scene = config.datasets[0][config.datasets[0].format].test_scene
        
    save_path = os.path.join(output_dir, "test", test_scene)
    common_utils.make_folder(os.path.dirname(save_path))
    
    if 'pretrained_path' in config.model:
        model.load_state_dict(torch.load(config.model.pretrained_path, 'cpu'))
    else:
        model.load_state_dict(torch.load(model_path + '.pt', 'cpu'))
        
    loss, losses = trainer.evaluate(data_loader_test, save_path=save_path)
    log('Test loss: %.7f, %s' % (loss,
        ', '.join(['%s: %.7f' % (k, v) for k, v in losses.items()])
    ))


def parse_args():
    parser = argparse.ArgumentParser(description='')

    # Exp config
    parser.add_argument('--exp_path', default='../exps', type=str)
    parser.add_argument('--exp', default='main', type=str)

    # Training-related
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--workers', default=0, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    suffix = ''
    args.suffix = suffix

    root_path = fullfile(args.exp_path, args.exp)
    os.makedirs(fullfile(root_path, 'logs'), exist_ok=True)
    initLogger(Logger(fullfile(root_path, 'logs/output%s.log' % (suffix)), overwrite=args.training and not args.recover))
    args.root_path = root_path

    main(args)
