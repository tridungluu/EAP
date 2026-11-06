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
from .distillers import FeatureDistillationManager

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


class Student_Processor(Processor):
    """
        Student: L_S = w_ce * CE + w_cd * CD + w_recon * Recon (Eq. 16)
    """

    def load_model(self):
        # teacher (dong bang)
        self.teacher_model = self.io.load_model(self.arg.teacher_model,
                                                **dict(self.arg.teacher_model_args))
        self.teacher_model.load_state_dict(
            torch.load(self.arg.teacher_weights, map_location='cpu')['model'])
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.to(self.dev)
        self.teacher_model.eval()

        # student
        model_args = dict(self.arg.model_args)
        model_args['feature_distillation'] = self.arg.feature_distillation
        self.model = self.io.load_model(self.arg.model, **model_args)
        self.model.apply(weights_init)
        # Fine-tune decoder cua teacher. Phai goi SAU weights_init.
        self.model.init_decoder_from_teacher(self.teacher_model)
        self.model.to(self.dev)
        self.loss = nn.CrossEntropyLoss()

        self.feat_distill_manager = FeatureDistillationManager(
            distill_config=self.arg.feature_distillation
        )

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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            cooldown=5,
            min_lr=1e-5
        )

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
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
        self.teacher_model.eval()
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []

        for data_teacher, data_student, label, _ in loader:

            data_teacher = data_teacher.float().to(self.dev)
            data_student = data_student.float().to(self.dev)
            label = label.long().to(self.dev)

            # teacher: chuoi day du -> Context
            with torch.no_grad():
                ctx_teacher = self.teacher_model(data_teacher)[1].detach()

            # student: chuoi quan sat mot phan -> logits, Context, chuoi tai tao
            output, ctx_student, recon_student = self.model(data_student)

            # Eq. 12: CE
            loss_ce = self.loss(output, label)

            # Eq. 15 (CD) + Eq. 14 (Recon, target = Partial Sequence cua student, Fig. 2a)
            loss_feat_distill, loss_dict = self.feat_distill_manager(
                ctx_student=ctx_student,
                ctx_teacher=ctx_teacher,
                recon_student=recon_student,
                target_skel=self.model._to_ctr_input(data_student).squeeze(-1)
            )

            # Eq. 16 (alpha dong vai tro w_ce)
            loss = self.arg.alpha * loss_ce + loss_feat_distill

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info['loss'] = loss.data.item()
            self.iter_info['ce'] = '{:.4f}'.format(loss_ce.item())
            self.iter_info['lr'] = '{:.6f}'.format(self.optimizer.param_groups[0]['lr'])
            self.iter_info.update({k: v.item() for k, v in loss_dict.items()})
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.scheduler.step(np.mean(loss_value))
        self.io.print_timer()

    def test(self, evaluation=True):
        self.model.eval()
        loader = self.data_loader['test']
        result_frag = []
        label_frag = []

        for _, data, label, _ in loader:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            with torch.no_grad():
                output = self.model(data)[0]          # logits; decoder khong chay o eval
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
            description='CTR-GCN Student (FC + CE + CD + Recon)')

        # region arguments yapf: disable
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser