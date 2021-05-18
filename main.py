import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import loader, processor as processor

warnings.filterwarnings('ignore')


base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')

model_path = os.path.join(base_path, 'models')
if not os.path.exists(model_path):
    os.mkdir(model_path)

parser = argparse.ArgumentParser(description='Text to Emotive Gestures Generation')
parser.add_argument('--dataset', type=str, default='mpi', metavar='D',
                    help='dataset to train or evaluate method (default: mpi)')
parser.add_argument('--frame-drop', type=int, default=2, metavar='FD',
                    help='frame down-sample rate (default: 2)')
parser.add_argument('--add-mirrored', type=bool, default=False, metavar='AM',
                    help='perform data augmentation by mirroring all the sequences (default: False)')
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: False)')
parser.add_argument('--use-multiple-gpus', type=bool, default=True, metavar='T',
                    help='use multiple GPUs if available (default: True)')
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
# parser.add_argument('--window-length', type=int, default=1, metavar='WL',
#                     help='max number of past time steps to take as input to transformer decoder (default: 60)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=1e-3, metavar='LR',
                    help='base learning rate (default: 1e-3)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: 0.05 * np.arange(20))')
parser.add_argument('--lr-decay', type=float, default=0.999, metavar='LRD',
                    help='learning rate decay (default: 0.999)')
parser.add_argument('--tf-decay', type=float, default=0.995, metavar='TFD',
                    help='teacher forcing ratio decay (default: 0.995)')
parser.add_argument('--gradient-clip', type=float, default=0.5, metavar='GC',
                    help='gradient clip threshold (default: 0.5) [not used in current version]')
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

# from sklearn.manifold import TSNE
# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# X_embedded = TSNE(n_components=2).fit_transform(X)
# plt.clf()
# for el in X_embedded:
#     plt.plot(el[0], el[1], marker='.', markersize=100)
# plt.show()

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


plot_aff = False
if plot_aff:
    os.makedirs('plots/intended', exist_ok=True)
    os.makedirs('plots/perceived', exist_ok=True)
    aff_descriptions = ['AN_R_C2_H', 'AN_RE_C4_H', 'AN_LE_C4_H', 'AN_R_LW_C4',
                        'AN_R_RW_C4', 'AN_LS_LE_LW', 'AN_RS_RE_RW',
                        'DR_LW_R_w_LW_C4', 'DR_LW_R_w_LW_H', 'DR_RW_R_w_RW_C4',
                        'DR_RW_R_w_RW_H', 'DR_LW_RW_w_R_H',
                        'AR_LW_R_RW_w_LW_C4_RW', 'AR_H_LW_R_w_H_RW_R', 'AR_C4_LW_R_w_C4_RW_R']
    aff_by_intended_emotion = [[] for _ in range(intended_emotion_dim)]
    aff_by_perceived_emotion = [[] for _ in range(intended_emotion_dim)]
    affs_dim_to_plot = affs_dim
    aff_max = -np.inf * np.ones(affs_dim_to_plot)
    aff_min = np.inf * np.ones(affs_dim_to_plot)
    colors = ['r', 'g', 'b']
    intended_color_by_nt = [[] for _ in range(intended_emotion_dim)]
    perceived_color_by_nt = [[] for _ in range(intended_emotion_dim)]
    intended_native_tongue = [[] for _ in range(intended_emotion_dim)]
    perceived_native_tongue = [[] for _ in range(intended_emotion_dim)]
    for key in data_dict.keys():
        for aff_idx in range(affs_dim_to_plot):
            if aff_max[aff_idx] < np.max(data_dict[key]['affective_features'][:, aff_idx]):
                aff_max[aff_idx] = np.max(data_dict[key]['affective_features'][:, aff_idx])
            if aff_min[aff_idx] > np.min(data_dict[key]['affective_features'][:, aff_idx]):
                aff_min[aff_idx] = np.min(data_dict[key]['affective_features'][:, aff_idx])
        intended_idx = np.where(data_dict[key]['Intended emotion'])[0][0]
        perceived_idx = np.where(data_dict[key]['Perceived category'])[0][0]
        aff_by_intended_emotion[intended_idx].append(data_dict[key]['affective_features'])
        intended_color_by_nt[intended_idx].append(colors[np.where(data_dict[key]['Native tongue'])[0][0]])
        intended_native_tongue[intended_idx].append(tag_categories[8][np.where(data_dict[key]['Native tongue'])[0][0]])
        aff_by_perceived_emotion[perceived_idx].append(
            data_dict[key]['affective_features'])
        perceived_color_by_nt[perceived_idx].append(
            colors[np.where(data_dict[key]['Native tongue'])[0][0]])
        perceived_native_tongue[perceived_idx].append(
            tag_categories[8][np.where(data_dict[key]['Native tongue'])[0][0]])

    def plot_aff_features(emo_category, aff_array, _native_tongues, nt_array):
        aff_means = []
        aff_stds = []
        for aff_idx in range(affs_dim_to_plot):
            aff_means.append([])
            aff_stds.append([])
            for emotion in range(len(aff_array)):
                aff_means[aff_idx].append(np.nan * np.ones(
                    (len(_native_tongues), len(max(aff_array[emotion], key=lambda x: len(x))))))
                aff_stds[aff_idx].append(np.nan * np.ones(
                    (len(_native_tongues), len(max(aff_array[emotion], key=lambda x: len(x))))))
                affs_curr = np.nan * np.ones((len(aff_array[emotion]), len(
                    max(aff_array[emotion], key=lambda x: len(x)))))
                for i, j in enumerate(aff_array[emotion]):
                    affs_curr[i][0:len(j[:, aff_idx])] = j[:, aff_idx]
                for nt_idx, nt in enumerate(_native_tongues):
                    aff_means[aff_idx][emotion][nt_idx] = np.nanmean(
                            affs_curr[[i for i, x in enumerate(nt_array[emotion]) if x == nt]], axis=0)
                    aff_stds[aff_idx][emotion][nt_idx] = np.nanstd(
                            affs_curr[[i for i, x in enumerate(nt_array[emotion]) if x == nt]], axis=0)

        for aff_idx in range(affs_dim_to_plot):
            for emotion in range(len(aff_array)):
                for nt_idx, nt in enumerate(_native_tongues):
                    plt.plot(aff_means[aff_idx][emotion][nt_idx])
                    plt.fill_between(np.arange(len(aff_stds[aff_idx][emotion][nt_idx])),
                                     aff_means[aff_idx][emotion][nt_idx] + aff_stds[aff_idx][emotion][nt_idx],
                                     aff_means[aff_idx][emotion][nt_idx] - aff_stds[aff_idx][emotion][nt_idx],
                                     alpha=0.4)
                plt.ylim([aff_min[aff_idx], aff_max[aff_idx]])
                plt.grid(True)
                plt.legend(_native_tongues)
                tag_idx = 0 if emo_category == 'intended' else 2
                plt.title('{}: {}'.format(tag_categories[tag_idx][emotion], aff_descriptions[aff_idx]))
                plt.savefig('plots/{}/{}_{:02d}.png'.format(emo_category, tag_categories[tag_idx][emotion], aff_idx),
                            bbox_inches='tight')
                plt.clf()

    native_tongues = ['Hindi', 'German']
    plot_aff_features('intended', aff_by_intended_emotion, native_tongues, intended_native_tongue)
    plot_aff_features('perceived', aff_by_perceived_emotion, native_tongues, perceived_native_tongue)

    # fig_intended, ax_intended = plt.subplots(intended_emotion_dim, affs_dim)
    # fig_perceived, ax_perceived = plt.subplots(intended_emotion_dim, affs_dim)
    # for emo_idx in range(intended_emotion_dim):
    #     for aff_idx in range(affs_dim_to_plot):
    #         for intended_idx, array in enumerate(aff_by_intended_emotion[emo_idx]):
    #             array_to_plot = np.copy(array[:, aff_idx])
    #             # if aff > 0:
    #             #     array_to_plot += aff_max[aff - 1]
    #             ax_intended[emo_idx][aff_idx].plot(array_to_plot,
    #                                                color=intended_color_by_nt[emo_idx][intended_idx],
    #                                                alpha=0.5,
    #                                                label=intended_native_tongue[emo_idx][intended_idx])
    #         ax_intended[emo_idx][aff_idx].set_xlabel('frame number')
    #         ax_intended[emo_idx][aff_idx].set_ylabel('feature value')
    #         ax_intended[emo_idx][aff_idx].grid(True)
    #         ax_intended[emo_idx][aff_idx].set_title('{}: {}'.format(tag_categories[0][emo_idx],
    #                                                                 aff_descriptions[aff_idx]))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # labels, ids = np.unique(labels, return_index=True)
    # handles = [handles[i] for i in ids]
    # plt.legend(handles, labels, loc='best')
    # plt.savefig('plots/intended.png')
    # plt.clf()


    # for emo_idx in range(intended_emotion_dim):
    #     for aff_idx in range(affs_dim_to_plot):
    #         for intended_idx, array in enumerate(aff_by_intended_emotion[emo_idx]):
    #             array_to_plot = np.copy(array[:, aff_idx])
    #             # if aff > 0:
    #             #     array_to_plot += aff_max[aff - 1]
    #             plt.plot(array_to_plot,
    #                      color=intended_color_by_nt[emo_idx][intended_idx], alpha=0.5,
    #                      label=intended_native_tongue[emo_idx][intended_idx])
    #         handles, labels = plt.gca().get_legend_handles_labels()
    #         labels, ids = np.unique(labels, return_index=True)
    #         handles = [handles[i] for i in ids]
    #         plt.xlabel('frame number')
    #         plt.ylabel('feature value')
    #         plt.grid(True)
    #         plt.legend(handles, labels, loc='best')
    #         title = '{}: {}'.format(tag_categories[0][emo_idx], aff_descriptions[aff_idx])
    #         plt.title(title)
    #         plt.savefig('plots/intended/{}_{:02d}.png'.format(tag_categories[0][emo_idx], aff_idx),
    #                     bbox_inches='tight')
    #         plt.clf()
    #
    #         for perceived_idx, array in enumerate(aff_by_perceived_emotion[emo_idx]):
    #             array_to_plot = np.copy(array[:, aff_idx])
    #             # if aff > 0:
    #             #     array_to_plot += aff_max[aff - 1]
    #             plt.plot(array_to_plot,
    #                      color=perceived_color_by_nt[emo_idx][perceived_idx], alpha=0.5,
    #                      label=perceived_native_tongue[emo_idx][perceived_idx])
    #         handles, labels = plt.gca().get_legend_handles_labels()
    #         labels, ids = np.unique(labels, return_index=True)
    #         handles = [handles[i] for i in ids]
    #         plt.xlabel('frame number')
    #         plt.ylabel('feature value')
    #         plt.grid(True)
    #         plt.legend(handles, labels, loc='best')
    #         title = '{}: {}'.format(tag_categories[0][emo_idx], aff_descriptions[aff_idx])
    #         plt.title(title)
    #         plt.savefig('plots/perceived/{}_{:02d}.png'.format(tag_categories[0][emo_idx], aff_idx),
    #                     bbox_inches='tight')
    #         plt.clf()

    with open('plots/intended.html', 'w') as inf:
        inf.write('<table>')
        for emo_idx in range(intended_emotion_dim):
            inf.write('<tr>')
            for aff_idx in range(affs_dim_to_plot):
                inf.write('<td><img src = "intended/{}_{:02d}.png"></td>'.format(tag_categories[0][emo_idx], aff_idx))
            inf.write('</tr>')
        inf.write('</table>')

    with open('plots/perceived.html', 'w') as inf:
        inf.write('<table>')
        for emo_idx in range(intended_emotion_dim):
            inf.write('<tr>')
            for aff_idx in range(affs_dim_to_plot):
                inf.write('<td><img src = "perceived/{}_{:02d}.png"></td>'.format(tag_categories[0][emo_idx], aff_idx))
            inf.write('</tr>')
        inf.write('</table>')

pr = processor.Processor(args, data_path, data_loader, text_length, num_frames + 2,
                         affs_dim, num_joints, coords, rots_dim, tag_categories,
                         intended_emotion_dim, intended_polarity_dim,
                         acting_task_dim, gender_dim, age_dim, handedness_dim, native_tongue_dim,
                         joint_names, joint_parents,
                         generate_while_train=False, save_path=base_path)

if args.train:
    pr.train()
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

pr.generate_motion(samples_to_generate=len(data_loader['test']), randomized=randomized, animations_as_videos=False)
