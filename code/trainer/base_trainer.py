import time
import copy
import math
import os
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util.logger import log
from util import model_utils
from util.warmup_multistep_lr import WarmupMultiStepLR


class BaseTrainer:
    def __init__(self, config, model, loss_fns):
        self.config = config
        self.model = model
        self.loss_fns = loss_fns
        self.device = config.device
        self.training_config = config.training
        
        # Initialize TensorBoard SummaryWriter
        log_dir = os.path.join(config.output_dir, 'tb_logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_step(self, samples):
        raise NotImplementedError

    def eval_step(self, samples, save_path=None):
        raise NotImplementedError

    def train(self, model_path, train_data, val_data, test_data=None, recover=False):
        best_model = copy.deepcopy(self.model)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.lr)

        warmup_iters = self.training_config.lr_warmup_epochs * len(train_data.dataset)
        lr_milestones = [len(train_data.dataset) * m for m in self.training_config.lr_milestones]
        lr_scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=self.training_config.lr_gamma, warmup_iters=warmup_iters, warmup_factor=1e-5)

        start_epoch = 0
        if recover:
            start_epoch = model_utils.load_checkpoint(model_path + '.ckp', 0, self.model, optimizer, lr_scheduler, opt=torch.optim)

        best_val_loss = math.inf
        epoch = self.training_config.epoch
        save_step = self.training_config.save_step if 'save_step' in self.training_config else 10
        max_steps = self.training_config.max_steps if 'max_steps' in self.training_config else 2000

        for e in range(start_epoch, epoch):
            st = time.time()

            loss_sum, train_losses = self.train_one_epoch(train_data, optimizer, lr_scheduler, e, max_steps)

            if val_data is not None:
                val_loss, val_all_losses = self.evaluate(val_data)
                # Select best model based on the metric of joints error or total loss
                val_metric = val_all_losses.get('joints', val_loss)
            else:
                val_loss = 0
                val_all_losses = {}
                val_metric = 0

            # Log to TensorBoard
            self.writer.add_scalar('Loss/Train_Total', loss_sum, e + 1)
            for k, v in train_losses.items():
                self.writer.add_scalar(f'Loss/Train_{k}', v, e + 1)
                
            self.writer.add_scalar('Loss/Val_Total', val_loss, e + 1)
            for k, v in val_all_losses.items():
                self.writer.add_scalar(f'Loss/Val_{k}', v, e + 1)

            if val_data is None or val_metric < best_val_loss:
                best_val_loss = val_metric
                log('saving model')
                best_model = copy.deepcopy(self.model)
                torch.save(self.model.state_dict(), model_path + '.pt')
            
            if e == 0 or (e + 1 + save_step) % save_step == 0:
                torch.save(self.model.state_dict(), model_path + f'-epoch{e + 1}' + '.pt')

            try:
                model_utils.save_checkpoint(model_path + '.ckp', e, self.model, optimizer, lr_scheduler)
            except Exception as exception:
                log(exception)

            log('[%d/%d] loss: %.7f, val_loss: %.7f, %s'
                % (e + 1, epoch, loss_sum, val_loss,
                ', '.join(['%s: %.7f' % (k, v) for k, v in val_all_losses.items()])
                ))
                
            if test_data is not None:
                test_loss, test_all_losses = self.evaluate(test_data)
                log('Test loss: %.7f, %s' % (test_loss,
                    ', '.join(['%s: %.7f' % (k, v) for k, v in test_all_losses.items()])
                ))
                self.writer.add_scalar('Loss/Test_Total', test_loss, e + 1)
                for k, v in test_all_losses.items():
                    self.writer.add_scalar(f'Loss/Test_{k}', v, e + 1)

            log('Current epoch run time: %.3f s' % (time.time() - st))
            torch.cuda.empty_cache()

        self.writer.close()
        return best_model

    def train_one_epoch(self, train_data, optimizer, lr_scheduler, epoch_idx, max_steps=2000):
        self.model.train()
        loss_sum = 0
        accumulated_losses = defaultdict(float)
        N = 0

        st = time.time()
        for step, samples in enumerate(train_data):
            optimizer.zero_grad()

            loss_total, losses, b_size = self.train_step(samples)
            loss_total.backward()
            optimizer.step()

            loss_sum += loss_total.item() * b_size
            for k, v in losses.items():
                accumulated_losses[k] += v * b_size
            N += b_size

            log('Epoch [%d/%d] Step %d total_loss: %.7f, %s' % (
                epoch_idx + 1, self.training_config.epoch, step,
                loss_total.item(),
                ', '.join(['%s: %.7f' % (k, v) for k, v in losses.items()])
            ))
            
            lr_scheduler.step()
            if step >= max_steps:
                break

        for k in accumulated_losses:
            accumulated_losses[k] /= N if N > 0 else 1

        return loss_sum / N if N > 0 else 0, accumulated_losses

    def evaluate(self, data_loader, save_path=None):
        loss_sum = 0
        all_losses = defaultdict(float)
        N = 0
        
        self.model.eval()
        with torch.no_grad():
            for step, samples in enumerate(data_loader):
                loss_total, losses, b_size = self.eval_step(samples, save_path)
                
                loss_sum += loss_total * b_size
                for k, v in losses.items():
                    all_losses[k] += v * b_size
                N += b_size

                if save_path is None and step >= 63:
                    break

        for k in all_losses.keys():
            all_losses[k] /= N if N > 0 else 1

        return loss_sum / N if N > 0 else 0, dict(all_losses)
