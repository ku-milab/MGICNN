

from time import localtime, strftime
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--data_path', type=str, default="/home/jsyoon/bck_nn/dataset/")
parser.add_argument('--summ_path_root', type=int, default="/Data3/jsyoon/bck_nn/")
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument("--maxfold", type=int, default=5)
parser.add_argument("--multistream_mode", type=int, default=0) #0-element(proposed), 1- concat, 2-1x1 comv
parser.add_argument("--model_mode", type=int, default=0) #0-proposed, 1-RI , 2-LR, 3-ZI, 4- ZO
parser.add_argument("--deploy", dest="deploy", action="store_true")
parser.add_argument("--devel", dest="deploy", action="store_false")
parser.add_argument("--GPU", type=int, default=-1) #0-proposed, 1-RI , 2-LR, 3-ZI, 4- ZO
parser.set_defaults(train=True)
parser.set_defaults(deploy=False)

ARGS = parser.parse_args()

deploy = ARGS.deploy
train= ARGS.train
fold_num = ARGS.fold
max_fold = ARGS.maxfold
multistream_mode = ARGS.multistream_mode
model_mode = ARGS.model_mode
set_gpu = ARGS.GPU
data_path = ARGS.data_path
summ_path_root = ARGS.summ_path_root


batch_norm = True
dropout = True
is_training = tf.placeholder(shape=[], dtype=tf.bool)

alpha = 0.2
beta1 = 0.5
beta2 = 1 - 1e-3
lr = 0.003

epoch = 100
batch_size = 128

summ_path = summ_path_root+"%s_%d_%d_%d/"%(strftime("%m%d_%H%M%S", localtime()), fold_num, multistream_mode, model_mode)