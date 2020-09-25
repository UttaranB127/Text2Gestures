import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import loader, processor as processor
from utils.visualizations import display_animations

warnings.filterwarnings("ignore")


base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')

model_path = os.path.join(base_path, 'models')
if not os.path.exists(model_path):
    os.mkdir(model_path)

parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--dataset', type=str, default='mpi', metavar='D',
                    help='dataset to train or evaluate method (default: mpi)')
parser.add_argument('--frame-drop', type=int, default=2, metavar='FD',
                    help='frame down-sample rate (default: 2)')
parser.add_argument('--add-mirrored', type=bool, default=False, metavar='AM',
                    help='perform data augmentation by mirroring all the sequences (default: False)')
parser.add_argument('--train', type=bool, default=False, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--use-multiple-gpus', type=bool, default=True, metavar='T',
                    help='use multiple GPUs if available (default: True)')
parser.add_argument('--load-last-best', type=bool, default=True, metavar='LB',
                    help='load the most recent best model (default: True)')
parser.add_argument('--batch-size', type=int, default=8, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='number of threads? (default: 4)')
parser.add_argument('--start-epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num-epoch', type=int, default=5000, metavar='NE',
                    help='number of epochs to train (default: 1000)')
# parser.add_argument('--window-length', type=int, default=1, metavar='WL',
#                     help='max number of past time steps to take as input to transformer decoder (default: 60)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=5e-3, metavar='LR',
                    help='base learning rate (default: 1e-3)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--lr-decay', type=float, default=0.999, metavar='LRD',
                    help='learning rate decay (default: 0.999)')
parser.add_argument('--tf-decay', type=float, default=0.995, metavar='TFD',
                    help='teacher forcing ratio decay (default: 0.995)')
parser.add_argument('--gradient-clip', type=float, default=0.5, metavar='GC',
                    help='gradient clip threshold (default: 0.1)')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--upper-body-weight', type=float, default=1., metavar='UBW',
                    help='loss weight on the upper body joint motions (default: 2.05)')
parser.add_argument('--affs-reg', type=float, default=0.8, metavar='AR',
                    help='regularization for affective features loss (default: 0.01)')
parser.add_argument('--quat-norm-reg', type=float, default=0.1, metavar='QNR',
                    help='regularization for unit norm constraint (default: 0.01)')
parser.add_argument('--quat-reg', type=float, default=1.2, metavar='QR',
                    help='regularization for quaternion loss (default: 0.01)')
parser.add_argument('--recons-reg', type=float, default=1.2, metavar='RCR',
                    help='regularization for reconstruction loss (default: 1.2)')
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
randomized = False

args.work_dir = os.path.join(model_path, args.dataset)
if not os.path.exists(args.work_dir):
    os.mkdir(args.work_dir)

data_dict, tag_categories, text_length, num_frames = loader.load_data(data_path, args.dataset,
                                                                      frame_drop=args.frame_drop,
                                                                      add_mirrored=args.add_mirrored)
data_dict_train, data_dict_eval = loader.split_data_dict(data_dict, randomized=False, fill=6)
any_dict_key = list(data_dict)[0]

# for key in list(data_dict.keys()):
#     aff = data_dict[key]['affective_features']
#     for idx in range(aff.shape[-1]):
#         plt.plot(aff[:, idx])
#         plt.title(str(data_dict[key]['Text']))
#         plt_dir = '../plots/aff_{:03d}'.format(idx)
#         os.makedirs(plt_dir, exist_ok=True)
#         plt.savefig(os.path.join(plt_dir, '{}_v_{:0.3f}_a_{:0.3f}_d_{:0.3f}.png'.format(
#             key,
#             data_dict[key]['Intended emotion VAD'][0],
#             data_dict[key]['Intended emotion VAD'][1],
#             data_dict[key]['Intended emotion VAD'][2])))
#         plt.clf()

from sklearn.manifold import TSNE
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2).fit_transform(X)
plt.clf()
for el in X_embedded:
    plt.plot(el[0], el[1], marker='.', markersize=100)
plt.show()

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

for emo_idx, emotion in enumerate(tag_categories[2]):
    data_points_intended = []
    data_points_perceived = []
    data_genders_intended = []
    data_genders_perceived = []
    for data_idx in range(len(data_dict_eval)):
        key = str(data_idx).zfill(6)
        if data_dict_eval[key]['Intended emotion'][emo_idx] == 1:
            data_points_intended.append(data_idx)
            data_genders_intended.append(tag_categories[5][np.where(data_dict_eval[key]['Gender'])[0][0]])
        if data_dict_eval[key]['Perceived category'][emo_idx] == 1:
            data_points_perceived.append(data_idx)
            data_genders_perceived.append(tag_categories[5][np.where(data_dict_eval[key]['Gender'])[0][0]])

# trim: 17, 20, 142
selected_female = [2, 6, 8, 12, 17, 20, 22, 27, 33, 34, 35, 37, 46, 50, 60, 62, 70,
                   75, 83, 88, 90, 94, 100, 102, 104, 116, 121, 125, 128, 130, 137, 142]
# trim: 4, 40, 56
selected_male = [0, 4, 5, 7, 14, 26, 32, 40, 41, 43, 49, 56, 57, 58, 66, 74, 78, 95,
                 97, 98, 101, 107, 110, 119, 120, 122, 124, 127, 132, 134, 138, 139]

selected_all = selected_female.copy()
selected_all += [m for m in selected_male]

choices_dir = os.path.join(base_path, 'study_multiple_choices')
os.makedirs(choices_dir, exist_ok=True)

shortlisted = [5, 14, 34, 50]
selected_all.sort()
for sl, sel_idx in enumerate(selected_all):
    key = str(sel_idx).zfill(6)
    text = data_dict_eval[key]['Text']
    gender = tag_categories[5][np.where(data_dict_eval[key]['Gender'])[0][0]]
    handedness = tag_categories[7][np.where(data_dict_eval[key]['Handedness'])[0][0]]
    intended_emotion = tag_categories[0][np.where(data_dict_eval[key]['Intended emotion'])[0][0]]
    perceived_emotion = tag_categories[2][np.where(data_dict_eval[key]['Perceived category'])[0][0]]
    print('{}\t{}\t{}\t{}\t{}\t{}'.format(sel_idx, intended_emotion[:3],
                                         perceived_emotion[:3], gender, handedness, text))
    other_emotions = set(tag_categories[0]) - {intended_emotion, perceived_emotion}

    if intended_emotion == perceived_emotion:
        selected_emotions = random.sample(list(other_emotions), k=3)
        selected_emotions.append(intended_emotion)
    else:
        selected_emotions = random.sample(list(other_emotions), k=2)
        selected_emotions.append(intended_emotion)
        selected_emotions.append(perceived_emotion)
    assert len(selected_emotions) == 4
    random.shuffle(selected_emotions)
    choices_file = '{}.txt'.format(key)
    with open(os.path.join(choices_dir, choices_file), 'w') as cf:
        for emo in selected_emotions:
            if emo == 'neutral':
                cf.write('Neutral\n')
            elif emo == 'joy':
                cf.write('Joyous\n')
            elif emo == 'shame':
                cf.write('Ashamed\n')
            elif emo == 'amusement':
                cf.write('Amused\n')
            elif emo == 'pride':
                cf.write('Proud\n')
            elif emo == 'sadness':
                cf.write('Sad\n')
            elif emo == 'surprise':
                cf.write('Surprised\n')
            elif emo == 'anger':
                cf.write('Angry\n')
            elif emo == 'fear':
                cf.write('Afraid\n')
            elif emo == 'relief':
                cf.write('Relieved\n')
            elif emo == 'disgust':
                cf.write('Disgusted\n')


pr = processor.Processor(args, data_path, data_loader, text_length, num_frames + 2,
                         affs_dim, num_joints, coords, rots_dim, tag_categories,
                         intended_emotion_dim, intended_polarity_dim,
                         acting_task_dim, gender_dim, age_dim, handedness_dim, native_tongue_dim,
                         joint_names, joint_parents,
                         generate_while_train=False, save_path=base_path)

# idx = 1302
# display_animations(np.swapaxes(np.reshape(
#     np.expand_dims(data_dict[str(idx)]['positions_world'], axis=0),
#     (1, num_frames, -1)), 2, 1), num_joints, coords, joint_parents,
#     save=True,
#     dataset_name=dataset, subset_name='test',
#     save_file_names=[str(idx)],
#     overwrite=True)

if args.train:
    pr.train()
# pr.generate_motion(data_dict_valid['0']['spline'], data_dict_valid['0'])
k = 0
index = str(k).zfill(6)
joint_offsets = torch.from_numpy(data_loader['test'][index]['joints_dict']['joints_offsets_all'][1:])
# pos = torch.from_numpy(data_loader[index]['positions'])
# affs = torch.from_numpy(data_loader[index]['affective_features'])
# quat = torch.cat((self.quats_sos,
#                   torch.from_numpy(data_loader[index]['rotations']),
#                   self.quats_eos), dim=0)
# quat_length = quat.shape[0]
# quat_valid_idx = torch.zeros(self.T)
# quat_valid_idx[:quat_length] = 1
# text = torch.cat((self.text_processor.numericalize(dataset[str(k).zfill(self.zfill)]['Text'])[0],
#                   torch.from_numpy(np.array([self.text_eos]))))
# if text[0] != self.text_sos:
#     text = torch.cat((torch.from_numpy(np.array([self.text_sos])), text))
# text_length = text.shape[0]
# text_valid_idx = torch.zeros(self.Z)
# text_valid_idx[:text_length] = 1

pr.generate_motion(samples_to_generate=len(data_loader['test']), randomized=randomized)
