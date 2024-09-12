import argparse
import random
import numpy as np
import torch
import os
from multi_runs import multiple_run
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.0005, type=float, help='(default=%(default)f)')
    parser.add_argument('--dataset', type=str, default='cifar10', help='(default=%(default)s)')
    parser.add_argument('--run_nums', type=int, default=10, help='(default=%(default)s)')
    parser.add_argument('--batch_size', type=int, default=10, help='(default=%(default)s)')


    parser.add_argument('--dataset_dir', type=str, default='/data/dataset/liqiwei/OCL/data/', help='(default=%(default)s)')
    parser.add_argument('--task_num', type=int, default=5, help='(default=%(default)s)')
    parser.add_argument('--alpha', type=float, default=2.65, help='(default=%(default)s)')
    parser.add_argument('--gpu_id', type=int, default=0, help='(default=%(default)s)')
    parser.add_argument('--n_workers', type=int, default=8, help='(default=%(default)s)')
    parser.add_argument('--log_dir', type=str, required=True, help='(default=%(default)s)')
    parser.add_argument('--epochs', type=int, default=1, help='(default=%(default)s)')
    parser.add_argument('--threshold', type=float, default=0.8, help='(default=%(default)s)')
    parser.add_argument('--optimi', type=str, default='Adam', help='(default=%(default)s)')
    parser.add_argument('--nf', type=int, default=20, help='(default=%(default)s)')
    parser.add_argument('--test_nme',  action='store_true',help="adopt lwf loss")
    parser.add_argument('--weight_con',  type=float, default=1, help='(default=%(default)s)')
    parser.add_argument('--prototypes_lr', type=float, default=35, help='(default=%(default)s)')
    parser.add_argument('--lr_factor', type=float, default=3, help='(default=%(default)s)')
    parser.add_argument('--miu',  type=float, default=1, help='(default=%(default)s)')
    parser.add_argument('--proj_gpm',  action='store_true',help="adopt lwf loss")
    parser.add_argument('--con_begin', type=int, default=0,help="layer gprompt")
    parser.add_argument('--proto_num', type=int, default=1,help="layer gprompt")
    parser.add_argument('--proto_ce', type=int, default=1000000,help="layer gprompt")
    parser.add_argument('--beta', type=float, default=2, help='(default=%(default)s)')
    parser.add_argument('--gamma', type=float, default=0.5, help='(default=%(default)s)')
    args = parser.parse_args()
    
    return args


def main(args):
    torch.cuda.set_device(args.gpu_id)
    args.cuda = torch.cuda.is_available()

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('[CUDA is unavailable]')
    
    multiple_run(args)


if __name__ == '__main__':
    args = get_params()
    main(args)
