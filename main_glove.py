import argparse
import os
import numpy as np

from utils import loader, processor_glove as processor

import warnings
warnings.filterwarnings('ignore')


base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')

model_path = os.path.join(base_path, 'models')
if not os.path.exists(model_path):
    os.mkdir(model_path)

parser = argparse.ArgumentParser(description='Text to Emotive Gestures Generation')
parser.add_argument('--dataset', type=str, default='mpi', metavar='D',
                    help='dataset to train or evaluate method (default: mpi)')
parser.add_argument('-embedding-src', default='glove.6B.300d.txt')
parser.add_argument('--frame-drop', type=int, default=2, metavar='FD',
                    help='frame down-sample rate (default: 2)')
parser.add_argument('--add-mirrored', type=bool, default=False, metavar='AM',
                    help='perform data augmentation by mirroring all the sequences (default: False)')
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--load-last-best', type=bool, default=True, metavar='LB',
                    help='load the most recent best model (default: True)')
parser.add_argument('--load-at-epoch', type=int, default=None, metavar='LAE',
                    help='load the model at the specified epoch (default: None)')
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 8)')
parser.add_argument('--start-epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num-epoch', type=int, default=5000, metavar='NE',
                    help='number of epochs to train (default: 5000)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=5e-3, metavar='LR',
                    help='base learning rate (default: 5e-3)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: 0.05 * np.arange(20))')
parser.add_argument('--lr-decay', type=float, default=0.999, metavar='LRD',
                    help='learning rate decay (default: 0.999)')
parser.add_argument('--tf-decay', type=float, default=0.995, metavar='TFD',
                    help='teacher forcing ratio decay (default: 0.995)')
parser.add_argument('--gradient-clip', type=float, default=0.5, metavar='GC',
                    help='gradient clip threshold (default: 0.5)')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--upper-body-weight', type=float, default=1., metavar='UBW',
                    help='loss weight on the upper body joint motions (default: 1.0)')
parser.add_argument('--affs-reg', type=float, default=0.8, metavar='AR',
                    help='regularization for affective features loss (default: 0.8)')
parser.add_argument('--quat-norm-reg', type=float, default=0.1, metavar='QNR',
                    help='regularization for unit norm constraint (default: 0.1)')
parser.add_argument('--quat-reg', type=float, default=1.2, metavar='QR',
                    help='regularization for quaternion loss (default: 1.2)')
parser.add_argument('--recons-reg', type=float, default=1.2, metavar='RCR',
                    help='regularization for reconstruction loss (default: 1.2)')
parser.add_argument('--min-train-epochs', type=int, default=20, metavar='MTE',
                    help='minimum number of training epochs after which the model'
                         'starts to get saved (default: 20)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--save-interval', type=int, default=10, metavar='SI',
                    help='interval after which model is saved (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'
randomized = False

args.work_dir = os.path.join(model_path, args.dataset + '_glove')
if not os.path.exists(args.work_dir):
    os.mkdir(args.work_dir)

data_dict, word2idx, embedding_table,\
    tag_categories, num_frames = loader.load_data_with_glove(data_path, args.dataset,
                                                             os.path.join(data_path, args.embedding_src),
                                                             frame_drop=args.frame_drop,
                                                             add_mirrored=args.add_mirrored)
data_dict_train, data_dict_eval = loader.split_data_dict(data_dict, randomized=False, fill=6)
any_dict_key = list(data_dict)[0]
affs_dim = data_dict[any_dict_key]['affective_features'].shape[-1]
num_joints = data_dict[any_dict_key]['positions'].shape[1]
coords = data_dict[any_dict_key]['positions'].shape[2]
joint_names = data_dict[any_dict_key]['joints_dict']['joints_names']
joint_parents = data_dict[any_dict_key]['joints_dict']['joints_parents']
data_loader = dict(train=data_dict_train, test=data_dict_eval)
prefix_length = int(0.3 * num_frames)
target_length = int(num_frames - prefix_length)
rots_dim = data_dict[any_dict_key]['rotations'].shape[-1]

intended_emotion_dim = data_dict[any_dict_key]['Intended emotion'].shape[-1]
intended_polarity_dim = data_dict[any_dict_key]['Intended polarity'].shape[-1]
acting_task_dim = data_dict[any_dict_key]['Acting task'].shape[-1]
gender_dim = data_dict[any_dict_key]['Gender'].shape[-1]
age_dim = 1
handedness_dim = data_dict[any_dict_key]['Handedness'].shape[-1]
native_tongue_dim = data_dict[any_dict_key]['Native tongue'].shape[-1]

pr = processor.Processor(args, data_path, data_loader, embedding_table.shape[-1], num_frames + 2,
                         affs_dim, num_joints, coords, rots_dim, tag_categories,
                         intended_emotion_dim, intended_polarity_dim,
                         acting_task_dim, gender_dim, age_dim, handedness_dim, native_tongue_dim,
                         joint_names, joint_parents, word2idx, embedding_table,
                         generate_while_train=True, save_path=base_path, device=device)

if args.train:
    pr.train()

pr.generate_motion(samples_to_generate=len(data_loader['test']), randomized=randomized, animations_as_videos=False)
