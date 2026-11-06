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

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO
import time

class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_data()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_optimizer()

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        

    def load_optimizer(self):
        pass

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()
        
    def print_networks(self, net, print_flag=False):
        self.io.print_log('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if print_flag:
            self.io.print_log(net)
        self.io.print_log('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        self.io.print_log('-----------------------------------------------')
        
  

    def measure_inference_time(self, num_warmup=10, num_iter=50):
        self.model.eval()
    
        loader = self.data_loader['test']
        device = next(self.model.parameters()).device
    
        times = []
    
        with torch.no_grad():
            # ---------- warm-up ----------
            for i, (data, label) in enumerate(loader):
                if i >= num_warmup:
                    break
                data = data.float().to(device)
                _ = self.model(data)
    
            # ---------- measure ----------
            for i, (data, label) in enumerate(loader):
                if i >= num_iter:
                    break
    
                data = data.float().to(device)
    
                if self.arg.use_gpu:
                    torch.cuda.synchronize()
    
                start = time.time()
                _ = self.model(data)
    
                if self.arg.use_gpu:
                    torch.cuda.synchronize()
    
                end = time.time()
                times.append(end - start)
    
        avg_time = np.mean(times)
        batch_size = data.shape[0]
    
        self.io.print_log(
            f'Inference time: {avg_time:.3f} s / batch | '
            f'{avg_time / batch_size:.3f} s / sample'
        )
    
        self.epoch_info['inference_ms_per_batch'] = avg_time 
        self.epoch_info['inference_ms_per_sample'] = avg_time / batch_size
    

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.print_networks(self.model)

        # training phase
        if self.arg.phase == 'train':
            self.best_result = 0.0
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.epoch = epoch
                self.meta_info['epoch'] = epoch
                
                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
#                if ((epoch + 1) % self.arg.save_interval == 0) or (
#                        epoch + 1 == self.arg.num_epoch):
#                    filename = 'epoch{}_model.pt'.format(epoch + 1)
#                    torch.save({
#                        'model': self.model.state_dict(),
#                        'adapter': self.text_encoder.adapter.state_dict(),
#                    }, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    self.test()
                    self.io.print_log("current %.2f%%, best %.2f%%" % 
                                            (self.current_result, self.best_result))
                     # save best model
                    filename = 'epoch%.3d_acc%.2f_model.pt' % (epoch + 1, self.current_result)
                    self.save_model(self.model, filename)
                    if self.current_result >= self.best_result:
                        filename = 'best_model.pt'
                        self.save_model(self.model, filename)
    

        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))
            self.best_result = 0.0

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            #self.measure_inference_time()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):

        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        #endregion yapf: enable
        
        # text encoder
        parser.add_argument('--text_encoder', default=None, help='the text encoder will be used')
        parser.add_argument('--text_args', action=DictAction, default=dict(), help='the arguments of text encoder')
        
        # teacher
        parser.add_argument('--teacher_model', default=None, help='the model will be used')
        parser.add_argument('--teacher_model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--teacher_weights', default=None, help='the weights for teacher network initialization')
        
        # KL distillation
        parser.add_argument('--alpha', default=None, help='weight of contrastive learning')
        parser.add_argument('--feature_distillation', action=DictAction, 
                   default=dict(cosine=0, align=0, spatial=1), 
                   help='feature distillation config: cosine, align, spatial weights')
        parser.add_argument('--logits_distillation', action=DictAction, 
                   default=dict(kd=0.5, hardkd=0, dkd=0, ctkd=0), 
                   help='logits distillation config: kd, hardkd, dkd, ctkd weights')
        parser.add_argument('--kd_T', type=float, default=1, help='default temperature for KD distillation')

        return parser
