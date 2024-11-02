import argparse
from .util import str2bool

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--work-dir',default='./work_dir/temp',help='the work folder for storing results')
    parser.add_argument('--config',default='./config/train.yaml',help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score',type=str2bool,default=False,help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker',type=int,default=8,help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args',default=dict(),help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args',default=dict(),help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args',type=dict,default=dict(),help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')
    parser.add_argument('--ignore-weights',type=str,default=[],nargs='+',help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--step',type=int,default=[20, 40, 60],nargs='+',help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device',type=str,default='cuda:0',help='the name of device for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch',type=int,default=0,help='start training from which epoch')
    parser.add_argument('--num-epoch',type=int,default=80,help='stop training in which epoch')
    parser.add_argument('--weight-decay',type=float,default=0.0005,help='weight decay for optimizer')

    return parser