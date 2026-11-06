#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Teacher_Processor(Processor):
    """
        Teacher: FC + CE, STA loss, DSTCL (Ranasingha et al., 2026), backbone CTR-GCN
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **dict(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        trainable_params = list(self.model.parameters())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (self.meta_info['epoch'] + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, _, label, _ in loader:

            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward: Context -> FC -> logits; Context -> Decoder -> recon; Latent -> (C_s, C_t)
            output, context, recon_skel, (C_s, C_t) = self.model(data)

            # Eq. 12: CE loss tren logits cua FC
            loss_ce = self.loss(output, label)

            # Eq. 11: STA loss (chuoi that vs chuoi tai tao)
            loss_sta, sample_sta = self.model.sta_loss(data, recon_skel)

            # Eq. 1, 8: DSTCL. TP/FP/FN va p_ik lay tu logits cua classifier
            loss_dstcl = self.model.dstcl_loss(C_s, C_t, label, output, sample_sta)

            # Eq. 13
            loss = (self.arg.w_sta * loss_sta
                    + self.arg.w_ce * loss_ce
                    + self.arg.w_cl * loss_dstcl)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['sta'] = '{:.4f}'.format(loss_sta.item())
            self.iter_info['ce'] = '{:.4f}'.format(loss_ce.item())
            self.iter_info['dstcl'] = '{:.4f}'.format(loss_dstcl.item())
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        result_frag = []
        label_frag = []

        for data, _, label, _ in loader:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            with torch.no_grad():
                output = self.model(data)[0]          # logits cua FC
            result_frag.append(output.data.cpu().numpy())

            if evaluation:
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.show_epoch_info()
            for k in self.arg.show_topk:
                self.show_topk(k)
            self.show_best(1)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='CTR-GCN Teacher (FC + CE + STA + DSTCL)')

        # region arguments yapf: disable
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--warm_up_epoch', type=int, default=0)
        # loss weights, Eq. 13
        parser.add_argument('--w_sta', type=float, default=1.0, help='weight of L_STA')
        parser.add_argument('--w_ce', type=float, default=1.0, help='weight of L_CE')
        parser.add_argument('--w_cl', type=float, default=0.1, help='weight of L_DSTCL')
        # endregion yapf: enable

        return parser