"""
Created by Wang Han on 2019/3/29 14:43.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""
import argparse

import torch

from utils.gpu import set_gpu
from utils.parse import parse_yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Radiology Segmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')
    parser.add_argument('--use_cuda', default='true', type=str,
                        help='whether use cuda. default: true')
    parser.add_argument('--use_parallel', default='false', type=str,
                        help='whether use parallel. default: false')
    parser.add_argument('--gpu', default='all', type=str,
                        help='use gpu device. default: all')
    parser.add_argument('--model', default='2D', type=str,
                        choices=['2D', '2.5D', '3D'],
                        help='which model used. default: 2D')
    parser.add_argument('--logdir', default=None, type=str,
                        help='which logdir used. default: None')
    parser.add_argument('--train_sample_csv', default=None, type=str,
                        help='train sample csv file used. default: None')
    parser.add_argument('--eval_sample_csv', default=None, type=str,
                        help='eval sample csv file used. default: None')
    parser.add_argument('--weight', default=None, type=str,
                        help='criterion weight. default: None')
    parser.add_argument('--mod_root', default=None, type=str,
                        help='modification dir used. default: None')
    parser.add_argument('--config', default='cfgs/seg2d/segment.yaml', type=str,
                        help='configuration file. default: cfgs/seg2d/segment.yaml')

    args = parser.parse_args()

    num_gpus = set_gpu(args.gpu)
    # set seed
    torch.manual_seed(args.seed)

    if args.model == '2D':
        from models.segment_model import Model
    elif args.model == '2.5D':
        from models.multi_segment_model import Model
    elif args.model == '3D':
        from models.segment_model_3d import Model

    config = parse_yaml(args.config)
    network_params = config['network']

    network_params['seed'] = args.seed
    network_params['device'] = "cuda" if str2bool(args.use_cuda) else "cpu"

    network_params['use_parallel'] = str2bool(args.use_parallel)
    network_params['num_gpus'] = num_gpus
    if num_gpus > 1:
        network_params['use_parallel'] = True
    config['network'] = network_params

    train_params = config['train']
    train_params['batch_size'] = train_params['batch_size'] * num_gpus
    train_params['num_workers'] = train_params['num_workers'] * num_gpus
    config['train'] = train_params

    eval_params = config['eval']
    eval_params['batch_size'] = eval_params['batch_size'] * num_gpus
    eval_params['num_workers'] = eval_params['num_workers'] * num_gpus
    config['eval'] = eval_params

    config['logging']['logging_dir'] = args.logdir if args.logdir is not None else config['logging']['logging_dir']
    try:
        if args.weight is not None:
            config['criterion']['cross_entropy_loss']['use_weight'] = True
            config['criterion']['cross_entropy_loss']['weight'] = [float(i) for i in args.weight.split(',')]
    except KeyError:
        pass

    data_params = config['data']
    data_params['train_sample_csv'] = args.train_sample_csv.split(',') if args.train_sample_csv is not None else \
        data_params['train_sample_csv']
    data_params['eval_sample_csv'] = args.eval_sample_csv.split(',') if args.eval_sample_csv is not None else \
        data_params['eval_sample_csv']
    if args.mod_root is not None:
        data_params['mod_root'] = args.mod_root
    config['data'] = data_params

    model = Model(config)
    if network_params['use_pretrained']:
        backbone = network_params['backbone'].split('_')[0]
        model.load_pretrained(network_params['pretrained_path'][backbone])
    model.run()
