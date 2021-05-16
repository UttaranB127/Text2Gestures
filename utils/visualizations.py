import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import os
import sys

from utils.mocap_dataset import MocapDataset


def display_animations(database, joint_parents, save=False, save_overlayed=False,
                       dataset_name=None, subset_name=None, save_file_names=None,
                       fill=6, overwrite=False):
    '''
    :param database: Should be of size N x VC x T
    :param joint_parents:
    :param save:
    :param save_overlayed:
    :param dataset_name:
    :param subset_name:
    :param save_file_names:
    :param fill:
    :param overwrite:
    :return:
    '''
    animations = []
    scale_max = -np.inf * np.ones(3)
    scale_min = np.inf * np.ones(3)

    for di, data in enumerate(database):
        num_frames = data.shape[-1]
        joints = np.swapaxes(np.squeeze(data), -1, 0).reshape((num_frames, -1, 3))
        animations.append(joints)
        max_curr = np.max(np.max(joints, axis=0), axis=0)
        min_curr = np.min(np.min(joints, axis=0), axis=0)
        if scale_min[0] > min_curr[0]:
            scale_min[0] = min_curr[0]
        if scale_min[1] > min_curr[1]:
            scale_min[1] = min_curr[1]
        if scale_min[2] > min_curr[2]:
            scale_min[2] = min_curr[2]
        if scale_max[0] < max_curr[0]:
            scale_max[0] = max_curr[0]
        if scale_max[1] < max_curr[1]:
            scale_max[1] = max_curr[1]
        if scale_max[2] < max_curr[2]:
            scale_max[2] = max_curr[2]
        print('\rProcessing joints: {0:d}/{1:d} done ({2:.2f}%).'
              .format(di+1, len(database), 100. * (di + 1) / len(database)), end='')
    plot_animations(animations, joint_parents, scale_max, scale_min,
                    save=save,
                    save_overlayed=save_overlayed,
                    dataset_name=dataset_name, subset_name=subset_name, save_file_names=save_file_names,
                    fill=fill, overwrite=overwrite)
    print()


def animate_joints(_i, _lines, _animations, _parents):
    num_animations = len(_animations) if isinstance(_animations, list) else 1
    if num_animations > 1:
        num_joints = int(len(_lines) / num_animations)
        for a in range(num_animations):
            for j in range(len(_parents)):
                if _parents[j] != -1:
                    _lines[a * num_joints + j].set_data(
                        [_animations[a][_i, j, 0], _animations[a][_i, _parents[j], 0]],
                        [-_animations[a][_i, j, 2], -_animations[a][_i, _parents[j], 2]])
                    _lines[a * num_joints + j].set_3d_properties(
                        [_animations[a][_i, j, 1], _animations[a][_i, _parents[j], 1]])
    elif num_animations == 1:
        for j in range(len(_parents)):
            if _parents[j] != -1:
                _lines[j].set_data(
                    np.asarray([_animations[_i, j, 0], _animations[_i, _parents[j], 0]]),
                    np.asarray([-_animations[_i, j, 2], -_animations[_i, _parents[j], 2]]))
                _lines[j].set_3d_properties(
                    [_animations[_i, j, 1], _animations[_i, _parents[j], 1]])
    return _lines


def plot_animations(animations, joint_parents, scale_max, scale_min,
                    save=False, save_overlayed=False,
                    dataset_name=None, subset_name=None, save_file_names=None,
                    fill=6, overwrite=False):
    if save:
        if not os.path.exists('videos'):
            os.makedirs('videos')
        dir_name = os.path.join('videos', dataset_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if subset_name is not None:
            dir_name = os.path.join(dir_name, subset_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
    total_animations = len(animations)
    joint_colors = list(sorted(colors.cnames.keys()))[::-1]
    total_colors = len(joint_colors)
    num_frames = len(animations[0])
    for ai, anim in enumerate(animations):
        if save:
            if save_file_names is None:
                save_file_name = os.path.join(dir_name, str(ai).zfill(fill) + '.mp4')
            else:
                save_file_name = os.path.join(dir_name, save_file_names[ai] + '.mp4')
            if not overwrite and os.path.exists(save_file_name):
                continue
        fig = plt.figure(figsize=(12, 8))
        ax = p3.Axes3D(fig)
        ax.plot3D(anim[:, 0, 0], -anim[:, 0, 2], np.zeros_like(anim[:, 0, 1]), 'gray')
        ax.set_xlim3d(scale_min[0], scale_max[0])
        ax.set_zlim3d(scale_min[1], scale_max[1])
        ax.set_ylim3d(scale_min[2], scale_max[2])
        # ax.set_xticks([], False)
        # ax.set_yticks([], False)
        # ax.set_zticks([], False)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.patch.set_alpha(0.)
        lines = [ax.plot([0, 0], [0, 0], [0, 0], color=joint_colors[ai % total_colors],
                         lw=2,
                         path_effects=[pe.Stroke(linewidth=3, foreground='black'),
                                       pe.Normal()])[0] for _ in range(anim.shape[1])]
        # ani = animation.FuncAnimation(fig, animate_joints, frames=num_frames,
        #                               fargs=(lines, animations[ai], parents),
        #                               interval=100, blit=True, repeat=True)
        if save:
            animation.FuncAnimation(fig, animate_joints, frames=num_frames,
                                    fargs=(lines, animations[ai], joint_parents),
                                    interval=100, blit=True, repeat=True).save(save_file_name)
        print('\rGenerating animations: {0:d}/{1:d} done ({2:.2f}%).'
              .format(ai + 1, total_animations, 100. * (ai + 1) / total_animations), end='')
        plt.cla()
        plt.close()

    if save_overlayed:
        save_file_name = os.path.join(dir_name, 'overlayed.mp4')
        if overwrite or not os.path.exists(save_file_name):
            lines = []
            fig = plt.figure(figsize=(12, 8))
            ax = p3.Axes3D(fig)
            ax.set_xlim3d(scale_min[0], scale_max[0])
            ax.set_zlim3d(scale_min[1], scale_max[1])
            ax.set_ylim3d(scale_min[2], scale_max[2])
            ax.set_xticks([], False)
            ax.set_yticks([], False)
            ax.set_zticks([], False)
            for ai, anim in enumerate(animations):
                lines.extend([ax.plot([0, 0], [0, 0], [0, 0], color=joint_colors[ai % total_colors],
                                      lw=2,
                                      path_effects=[pe.Stroke(linewidth=3, foreground='black'),
                                                    pe.Normal()])[0] for _ in range(anim.shape[1])])
            animation.FuncAnimation(fig, animate_joints, frames=num_frames,
                                    fargs=(lines, animations, joint_parents),
                                    interval=100, blit=True, repeat=True).save(save_file_name)
    # else:
    #     ani = []
    #     for ai in range(len(animations)):
    #         ani.append(animation.FuncAnimation(fig[ai], animate_joints, frames=len(animations[ai]),
    #                                            fargs=(lines[ai], animations[ai], parents),
    #                                            interval=100, blit=True, repeat=True))
