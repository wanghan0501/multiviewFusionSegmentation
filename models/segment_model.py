"""
Created by Wang Han on 2019/3/29 11:29.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

import datasets.transforms.segment_transforms as ST
from datasets.segment_dataset import SegmentDataset
from nets.cores.meter import AverageMeter
from nets.net_selector import NetSelector
from utils.log import get_logger
from utils.parse import format_config


class Model:

    def __init__(self, config):
        self.config = config
        # loading network parameters
        network_params = config['network']
        self.device = torch.device(network_params['device'])
        self.net = NetSelector(config).get_net()

        if network_params['use_parallel']:
            self.net = nn.DataParallel(self.net)

        self.net = self.net.to(self.device)
        self.epochs = config['optim']['num_epochs']

        # loading logging parameters
        run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
        logging_params = config['logging']
        if logging_params['logging_dir'] is None:
            self.ckpt_path = os.path.join(logging_params['ckpt_path'], run_timestamp)
        else:
            self.ckpt_path = os.path.join(logging_params['ckpt_path'], logging_params['logging_dir'])
        if logging_params['use_logging']:
            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.logger = get_logger(os.path.join(self.ckpt_path, '{}.log'.format(network_params['net_name'])))
            self.logger.info(">>>The config is:")
            self.logger.info(format_config(config))
            self.logger.info(">>>The net is:")
            self.logger.info(self.net)
        if logging_params['use_tensorboard']:
            from torch.utils.tensorboard import SummaryWriter

            if logging_params['logging_dir'] is None:
                self.run_path = os.path.join(logging_params['run_path'], run_timestamp)
            else:
                self.run_path = os.path.join(logging_params['run_path'], logging_params['logging_dir'])
            if not os.path.exists(self.run_path):
                os.makedirs(self.run_path)
            self.writer = SummaryWriter(self.run_path)

    def run(self):
        # optimizer
        optim_params = self.config['optim']
        if optim_params['optim_method'] == 'sgd':
            sgd_params = optim_params['sgd']
            optimizer = optim.SGD(
                self.net.parameters(),
                lr=sgd_params['base_lr'],
                weight_decay=sgd_params['weight_decay'],
                momentum=sgd_params['momentum'],
                nesterov=sgd_params['nesterov'])
        elif optim_params['optim_method'] == 'adam':
            adam_params = optim_params['adam']
            optimizer = optim.Adam(
                self.net.parameters(),
                lr=adam_params['base_lr'],
                betas=adam_params['betas'],
                weight_decay=adam_params['weight_decay'],
                amsgrad=adam_params['amsgrad'])
        elif optim_params['optim_method'] == 'adamw':
            adamw_params = optim_params['adamw']
            optimizer = optim.AdamW(
                self.net.parameters(),
                lr=adamw_params['base_lr'],
                betas=adamw_params['betas'],
                weight_decay=adamw_params['weight_decay'],
                amsgrad=adamw_params['amsgrad'])
        else:
            raise Exception('Not support optim method: {}.'.format(optim_params['optim_method']))

        # choosing whether to use lr_decay and related parameters
        lr_decay = None
        if optim_params['use_lr_decay']:
            from torch.optim import lr_scheduler
            if optim_params['lr_decay_method'] == 'cosine':
                cosine_params = optim_params['cosine']
                lr_decay = lr_scheduler.CosineAnnealingLR(
                    optimizer, eta_min=cosine_params['eta_min'], T_max=cosine_params['T_max'])
            elif optim_params['lr_decay_method'] == 'exponent':
                exponent_params = optim_params['exponent']
                lr_decay = lr_scheduler.ExponentialLR(
                    optimizer, gamma=exponent_params['gamma'])
            elif optim_params['lr_decay_method'] == 'warmup':
                warmup_params = optim_params['warmup']
                from target_segmentation.nets.cores.warmup_scheduler import GradualWarmupScheduler
                if warmup_params['after_scheduler'] == 'cosine':
                    cosine_params = optim_params['cosine']
                    after_scheduler = lr_scheduler.CosineAnnealingLR(
                        optimizer, eta_min=cosine_params['eta_min'], T_max=cosine_params['T_max'])
                elif warmup_params['after_scheduler'] == 'exponent':
                    exponent_params = optim_params['exponent']
                    after_scheduler = lr_scheduler.ExponentialLR(
                        optimizer, gamma=exponent_params['gamma'])
                else:
                    raise Exception('Not support after_scheduler method: {}.'.format(warmup_params['after_scheduler']))

                lr_decay = GradualWarmupScheduler(optimizer, multiplier=warmup_params['multiplier'],
                                                  total_epoch=warmup_params['total_epoch'],
                                                  after_scheduler=after_scheduler)

        data_params = self.config['data']
        # making train dataset and dataloader
        train_params = self.config['train']
        train_trans_seq = self._resolve_transforms(train_params['aug_trans'])
        train_dataset = SegmentDataset(
            data_root=data_params['data_root'],
            sample_records=data_params['train_sample_csv'],
            transforms=train_trans_seq,
            img_w=data_params['image_width'],
            img_h=data_params['image_height'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params['num_workers'],
            drop_last=True,
            pin_memory=train_params['pin_memory'])

        # making eval dataset and dataloader
        eval_params = self.config['eval']
        eval_trans_seq = self._resolve_transforms(eval_params['aug_trans'])
        eval_dataset = SegmentDataset(
            data_root=data_params['data_root'],
            sample_records=data_params['eval_sample_csv'],
            transforms=eval_trans_seq,
            img_w=data_params['image_width'],
            img_h=data_params['image_height'])
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_params['batch_size'],
            shuffle=False,
            num_workers=eval_params['num_workers'],
            pin_memory=eval_params['pin_memory'])

        # choosing criterion
        criterion_params = self.config['criterion']
        if criterion_params['criterion_method'] == 'cross_entropy_loss':
            from torch.nn import CrossEntropyLoss

            loss_params = criterion_params['cross_entropy_loss']
            if loss_params['use_weight']:
                weight = torch.Tensor(loss_params['weight'])
                criterion = CrossEntropyLoss(weight).to(self.device)
            else:
                criterion = CrossEntropyLoss().to(self.device)
        else:
            raise Exception('Not support criterion method: {}.'
                            .format(criterion_params['criterion_method']))

        # recording the best model
        best_dice = 0
        best_epoch = 0
        for epoch_id in range(optim_params['num_epochs']):
            self._train(epoch_id, train_loader, criterion, optimizer)
            if optim_params['use_lr_decay']:
                lr_decay.step()
            eval_dice_metric = self._eval(epoch_id, eval_loader, criterion)
            eval_dice = np.mean(eval_dice_metric[:, 0])

            # saving the best model
            if eval_dice >= best_dice:
                best_dice = eval_dice
                best_epoch = epoch_id
                self.save(epoch_id)
            self.logger.info('[Info] The maximal dice is {:.4f} at epoch {}'.format(
                best_dice,
                best_epoch))

    def _train(self, epoch_id, data_loader, criterion, optimizer):
        loss_meter = AverageMeter()
        front_classes = self.config['data']['num_classes'] - 1
        metric_idxs = [i + 1 for i in range(front_classes)]
        dice_volume_meter = np.zeros([front_classes, 4])
        self.net.train()
        with tqdm(total=len(data_loader)) as pbar:
            for batch_id, sample in enumerate(data_loader):
                image, label = sample['image'].to(self.device), sample['label'].to(self.device).long()

                optimizer.zero_grad()
                logits = self.net(image)

                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.data.item(), image.size(0))

                dice_volume = self.segment_volume(logits, label, metric_idxs)
                dice_volume_meter += dice_volume

                pbar.update(1)
                pbar.set_description("[Train] Epoch:{}, Loss:{:.4f}".format(
                    epoch_id, loss_meter.avg))

            dice_metrics = self.segment_metrics(dice_volume_meter)

        logging_params = self.config['logging']
        if logging_params['use_logging']:
            self.logger.info("[Train] Epoch:{}, Loss:{:.4f}, Dice Metric:\n{}"
                             .format(epoch_id, loss_meter.avg, dice_metrics))
        if logging_params['use_tensorboard']:
            self.writer.add_scalar('train/loss', loss_meter.avg, epoch_id)
            for i in range(front_classes):
                self.writer.add_scalar('train/dice_{}'.format(i), dice_metrics[i, 0], epoch_id)
                self.writer.add_scalar('train/TPVF_{}'.format(i), dice_metrics[i, 1], epoch_id)
                self.writer.add_scalar('train/PPV_{}'.format(i), dice_metrics[i, 2], epoch_id)

    def _eval(self, epoch_id, data_loader, criterion):
        loss_meter = AverageMeter()
        front_classes = self.config['data']['num_classes'] - 1
        metric_idxs = [i + 1 for i in range(front_classes)]
        dice_volume_meter = np.zeros([front_classes, 4])
        self.net.eval()
        with torch.no_grad():
            with tqdm(total=len(data_loader)) as pbar:
                for batch_id, sample in enumerate(data_loader):
                    image, label = sample['image'].to(self.device), sample['label'].to(self.device).long()

                    logits = self.net(image)

                    loss = criterion(logits, label)
                    loss_meter.update(loss.data.item(), image.size(0))

                    dice_volume = self.segment_volume(logits, label, metric_idxs)
                    dice_volume_meter += dice_volume

                    pbar.update(1)
                    pbar.set_description("[Eval] Epoch:{}, Loss:{:.4f}".format(
                        epoch_id, loss_meter.avg))

            dice_metrics = self.segment_metrics(dice_volume_meter)

            logging_params = self.config['logging']
            if logging_params['use_logging']:
                self.logger.info("[Eval] Epoch:{}, Loss:{:.4f}, Dice Metric:\n{}"
                                 .format(epoch_id, loss_meter.avg, dice_metrics))
            if logging_params['use_tensorboard']:
                self.writer.add_scalar('Eval/loss', loss_meter.avg, epoch_id)
                for i in range(front_classes):
                    self.writer.add_scalar('Eval/dice_{}'.format(i), dice_metrics[i, 0], epoch_id)
                    self.writer.add_scalar('Eval/TPVF_{}'.format(i), dice_metrics[i, 1], epoch_id)
                    self.writer.add_scalar('Eval/PPV_{}'.format(i), dice_metrics[i, 2], epoch_id)

        return dice_metrics

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.config['network']['device']))
        if self.config['network']['use_parallel']:
            self.net.module.load_state_dict(ckpt)
        else:
            self.net.load_state_dict(ckpt)
        print(">>> Loading model successfully from {}.".format(ckpt_path))

    def save(self, epoch):
        if self.config['network']['use_parallel']:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, os.path.join(self.ckpt_path, '{}.pth'.format(epoch)))

    def _resolve_transforms(self, aug_trans_params):
        """
            According to the given parameters, resolving transform methods
        :param aug_trans_params: the json of transform methods used
        :return: the list of augment transform methods
        """
        trans_seq = []
        for trans_name in aug_trans_params['trans_seq']:
            if trans_name == 'to_tensor':
                trans_seq.append(ST.ToTensor())
            elif trans_name == 'random_rotate':
                params = aug_trans_params['random_rotate']
                trans_seq.append(ST.RandomRotate(params['degree']))
            elif trans_name == 'scale':
                params = aug_trans_params['scale']
                trans_seq.append(ST.Scale(params['size']))
            elif trans_name == 'random_scale':
                params = aug_trans_params['random_scale']
                trans_seq.append(ST.RandomScale(params['limit']))
            elif trans_name == 'filter_and_normalize_by_WL':
                params = aug_trans_params['WL']
                trans_seq.append(ST.FilterAndNormalizeByWL(params['W'], params['L']))
            elif trans_name == 'arr2image':
                trans_seq.append(ST.Arr2Image())
            elif trans_name == 'random_horizontal_flip':
                trans_seq.append(ST.RandomHorizontalFlip())
            elif trans_name == 'random_sized':
                params = aug_trans_params['random_sized']
                trans_seq.append(
                    ST.RandomSized(params['size'], params['scale_min'], params['scale_max']))
            elif trans_name == 'center_crop':
                params = aug_trans_params['center_crop']
                trans_seq.append(ST.CenterCrop(params['size']))
            else:
                raise Exception('Not support transform method: {}.'.format(trans_name))

        return Compose(trans_seq)

    def load_pretrained(self, pretrained_path):
        network_params = self.config['network']
        backbone = network_params['backbone'].split('_')[0]

        if 'densenet' in backbone:
            pretrain_dict = torch.load(pretrained_path)

            if network_params['use_parallel']:
                model_dict = self.net.module.state_dict()
            else:
                model_dict = self.net.state_dict()

            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            pattern2 = re.compile(r'(?!classifier|features.conv0)')

            for key in list(pretrain_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = "backbone" + res.group(1)[8:] + res.group(2)
                    if new_key in model_dict.keys():
                        print("Loading parameter {}".format(new_key))
                        model_dict[new_key] = pretrain_dict[key]
                else:
                    res = pattern2.match(key)
                    if res:
                        new_key = "backbone" + key[8:]
                        if new_key in model_dict.keys():
                            print("Loading parameter {}".format(new_key))
                            model_dict[new_key] = pretrain_dict[key]

            if network_params['use_parallel']:
                self.net.module.load_state_dict(model_dict)
            else:
                self.net.load_state_dict(model_dict)
        elif 'resnet' in backbone:
            pretrain_dict = torch.load(pretrained_path)

            if network_params['use_parallel']:
                model_dict = self.net.module.state_dict()
            else:
                model_dict = self.net.state_dict()

            pattern = re.compile(r'(?!fc|conv1)')
            # import pdb;pdb.set_trace()
            for key in list(pretrain_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = "backbone." + key
                    if new_key in model_dict.keys():
                        print("Loading parameter {}".format(new_key))
                        model_dict[new_key] = pretrain_dict[key]

            if network_params['use_parallel']:
                self.net.module.load_state_dict(model_dict)
            else:
                self.net.load_state_dict(model_dict)
        elif 'vgg' in backbone:
            pretrain_dict = torch.load(pretrained_path)

            if network_params['use_parallel']:
                model_dict = self.net.module.state_dict()
            else:
                model_dict = self.net.state_dict()

            pattern = re.compile(r'(?!classifier|features.0)')
            # import pdb;pdb.set_trace()
            for key in list(pretrain_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = "backbone." + key
                    if new_key in model_dict.keys():
                        print("Loading parameter {}".format(new_key))
                        model_dict[new_key] = pretrain_dict[key]

            if network_params['use_parallel']:
                self.net.module.load_state_dict(model_dict)
            else:
                self.net.load_state_dict(model_dict)
        else:
            raise Exception('Not support pretrained backbone: {}.'.
                            format(network_params['backbone']))

    @staticmethod
    def accuracy(logits, target):
        """
        Computing the accuracy metric
        :param logits: the output of the given samples after network
        :param target:  the label of the given samples
        :return:
        """
        batch = logits.size(0)
        acc = target.eq(torch.argmax(logits, dim=1)).sum().cpu().item() / batch
        return acc

    @staticmethod
    def dice_by_batch(logits, labels, metric_idxs):
        dices = []
        probs = torch.argmax(logits, dim=1).data
        for metric_idx in metric_idxs:
            mask_o = (probs == metric_idx)
            mask_y = (labels == metric_idx)
            inter = (mask_o * mask_y).sum().float()
            union = mask_o.sum() + mask_y.sum()

            if inter == 0:
                dice = 0
            else:
                dice = ((2 * inter) / union).item()
            dices.append(dice)
        return dices

    @staticmethod
    def segment_volume(logits, labels, metric_idxs):
        volumes = []
        probs = torch.argmax(logits, dim=1).data
        for metric_idx in metric_idxs:
            mask_o = (probs == metric_idx)
            mask_y = (labels == metric_idx)
            inter = (mask_o * mask_y).sum()
            union = mask_o.sum() + mask_y.sum()
            v_y = mask_y.sum()
            v_o = mask_o.sum()
            volumes.append(np.array(
                [union.data.cpu().numpy(), inter.data.cpu().numpy(), v_y.data.cpu().numpy(), v_o.data.cpu().numpy()]))
        return np.array(volumes)

    @staticmethod
    def segment_metrics(volume):
        front_classes = volume.shape[0]
        metrics = []
        for class_idx in range(front_classes):
            union, inter, v_y, v_o = volume[class_idx, 0], volume[class_idx, 1], volume[class_idx, 2], volume[
                class_idx, 3]
            dice = 0 if union == 0 else float(2 * inter) / union
            TPVF = 0 if v_y == 0 else float(inter) / v_y
            PPV = 0 if v_o == 0 else float(inter) / v_o
            metrics.append(np.array([dice, TPVF, PPV]))
        return np.array(metrics)
