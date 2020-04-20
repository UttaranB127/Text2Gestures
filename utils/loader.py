# sys
import csv
import glob
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from utils.mocap_dataset import MocapDataset

# torch
import torch


def split_data_dict(data_dict, eval_size=0.1, randomized=True, fill=1):
    num_samples = len(data_dict)
    num_samples_valid = int(round(eval_size * num_samples))
    samples_all = np.array(list(data_dict.keys()), dtype=int)
    if randomized:
        samples_valid = np.random.choice(samples_all, num_samples_valid, replace=False)
    else:
        samples_valid = samples_all[-num_samples_valid:]
    samples_train = np.setdiff1d(samples_all, samples_valid)
    data_dict_train = dict()
    data_dict_eval = dict()
    for idx, sample_idx in enumerate(samples_train):
        data_dict_train[str(idx).zfill(fill)] = data_dict[str(sample_idx).zfill(fill)]
    for idx, sample_idx in enumerate(samples_valid):
        data_dict_eval[str(idx).zfill(fill)] = data_dict[str(sample_idx).zfill(fill)]
    return data_dict_train, data_dict_eval


def to_one_hot(categorical_value, categories):
    index = categories.index(categorical_value)
    one_hot_array = np.zeros(len(categories))
    one_hot_array[index] = 1.
    return one_hot_array


def load_data(_path, dataset, frame_drop=1, add_mirrored=False):
    data_path = os.path.join(_path, dataset)
    data_dict_file = os.path.join(data_path, 'data_dict_drop_' + str(frame_drop) + '.npz')
    try:
        data_dict = np.load(data_dict_file, allow_pickle=True)['data_dict'].item()
        max_text_length = np.load(data_dict_file, allow_pickle=True)['max_text_length'].item()
        max_time_steps = np.load(data_dict_file, allow_pickle=True)['max_time_steps'].item()
        print('Data file found. Returning data.')
    except FileNotFoundError:
        data_dict = []
        max_text_length = 0.
        max_time_steps = 0.
        if dataset == 'mpi':
            channel_map = {
                'Xrotation': 'x',
                'Yrotation': 'y',
                'Zrotation': 'z'
            }
            data_dict = dict()
            tag_names = []
            with open(os.path.join(data_path, 'tag_names.txt')) as names_file:
                for line in names_file.readlines():
                    line = line[:-1]
                    tag_names.append(line)
            id = tag_names.index('ID')
            relevant_tags = ['Intended emotion', 'Intended polarity',
                             'Perceived category', 'Perceived polarity',
                             'Acting task', 'Gender', 'Age', 'Handedness', 'Native tongue', 'Text']
            tag_categories = [[] for _ in range(len(relevant_tags) - 1)]
            tag_files = glob.glob(os.path.join(data_path, 'tags/*.txt'))
            num_files = len(tag_files)
            for tag_file in tag_files:
                tag_data = []
                with open(tag_file) as f:
                    for line in f.readlines():
                        line = line[:-1]
                        tag_data.append(line)
                for category in range(len(tag_categories)):
                    tag_to_append = relevant_tags[category]
                    if tag_data[tag_names.index(tag_to_append)] not in tag_categories[category]:
                        tag_categories[category].append(tag_data[tag_names.index(tag_to_append)])

            for data_counter, tag_file in enumerate(tag_files):
                tag_data = []
                with open(tag_file) as f:
                    for line in f.readlines():
                        line = line[:-1]
                        tag_data.append(line)
                bvh_file = os.path.join(data_path, 'bvh/' + tag_data[id] + '.bvh')
                names, parents, offsets,\
                positions, rotations = MocapDataset.load_bvh(bvh_file, channel_map)
                positions_down_sampled = positions[1::frame_drop]
                rotations_down_sampled = rotations[1::frame_drop]
                if len(positions_down_sampled) > max_time_steps:
                    max_time_steps = len(positions_down_sampled)
                joints_dict = dict()
                joints_dict['joints_to_model'] = np.arange(len(parents))
                joints_dict['joints_parents_all'] = parents
                joints_dict['joints_parents'] = parents
                joints_dict['joints_names_all'] = names
                joints_dict['joints_names'] = names
                joints_dict['joints_offsets_all'] = offsets
                joints_dict['joints_left'] = [idx for idx, name in enumerate(names) if 'left' in name.lower()]
                joints_dict['joints_right'] = [idx for idx, name in enumerate(names) if 'right' in name.lower()]
                data_dict[tag_data[id]] = dict()
                data_dict[tag_data[id]]['joints_dict'] = joints_dict
                data_dict[tag_data[id]]['positions'] = positions_down_sampled
                data_dict[tag_data[id]]['rotations'] = rotations_down_sampled
                data_dict[tag_data[id]]['affective_features'] =\
                    MocapDataset.get_mpi_affective_features(positions_down_sampled)
                for tag_index, tag_name in enumerate(relevant_tags):
                    if tag_name.lower() == 'text':
                        data_dict[tag_data[id]][tag_name] = tag_data[tag_names.index(tag_name)]
                        text_length = len(data_dict[tag_data[id]][tag_name])
                        if text_length > max_text_length:
                            max_text_length = text_length
                        continue
                    if tag_name.lower() == 'age':
                        data_dict[tag_data[id]][tag_name] = float(tag_data[tag_names.index(tag_name)]) / 100.
                        continue
                    if tag_name is 'Perceived category':
                        categories = tag_categories[0]
                    elif tag_name is 'Perceived polarity':
                        categories = tag_categories[1]
                    else:
                        categories = tag_categories[tag_index]
                    data_dict[tag_data[id]][tag_name] = to_one_hot(tag_data[tag_names.index(tag_name)], categories)
                print('\rData file not found. Processing file {}/{}: {:3.2f}%'.format(
                    data_counter + 1, num_files, data_counter * 100. / num_files), end='')
            print('\rData file not found. Processing files: done. Saving...', end='')
            np.savez_compressed(data_dict_file,
                                data_dict=data_dict,
                                max_text_length=max_text_length,
                                max_time_steps=max_time_steps)
            print('done. Returning data.')
        elif dataset == 'creative_it':
            mocap_data_dirs = os.listdir(os.path.join(data_path, 'mocap'))
            for mocap_dir in mocap_data_dirs:
                mocap_data_files = glob.glob(os.path.join(data_path, 'mocap/' + mocap_dir + '/*.txt'))
        else:
            raise FileNotFoundError('Dataset not found.')

    return data_dict, max_text_length, max_time_steps


def load_edin_labels(_path, num_labels):
    labels_dirs = [os.path.join(_path, 'labels_edin_locomotion_train')]
                   # os.path.join(_path, 'labels_edin_xsens')]
    labels = []
    num_annotators = np.zeros(len(labels_dirs))
    for didx, labels_dir in enumerate(labels_dirs):
        annotators = os.listdir(labels_dir)
        num_annotators_curr = len(annotators)
        labels_curr = np.zeros((num_labels[didx], num_annotators_curr))
        for file in annotators:
            with open(os.path.join(labels_dir, file)) as csv_file:
                read_line = csv.reader(csv_file, delimiter=',')
                row_count = -1
                for row in read_line:
                    row_count += 1
                    if row_count == 0:
                        continue
                    try:
                        data_idx = int(row[0].split('_')[-1])
                    except ValueError:
                        data_idx = row[0].split('/')[-1]
                        data_idx = int(data_idx.split('.')[0])
                    emotion = row[1].split(sep=' ')
                    behavior = row[2].split(sep=' ')
                    personality = row[3].split(sep=' ')
                    if len(emotion) == 1 and emotion[0].lower() == 'neutral':
                        labels_curr[data_idx, 3] += 1.
                    elif len(emotion) > 1:
                        counter = 0.
                        if emotion[0].lower() == 'extremely':
                            counter = 1.
                        elif emotion[0].lower() == 'somewhat':
                            counter = 1.
                        if emotion[1].lower() == 'happy':
                            labels_curr[data_idx, 0] += counter
                        elif emotion[1].lower() == 'sad':
                            labels_curr[data_idx, 1] += counter
                        elif emotion[1].lower() == 'angry':
                            labels_curr[data_idx, 2] += counter
                    if len(behavior) == 1 and behavior[0].lower() == 'neutral':
                        labels_curr[data_idx, 6] += 1.
                    elif len(behavior) > 1:
                        counter = 0.
                        if behavior[0].lower() == 'highly':
                            counter = 2.
                        elif behavior[0].lower() == 'somewhat':
                            counter = 1.
                        if behavior[1].lower() == 'dominant':
                            labels_curr[data_idx, 4] += counter
                        elif behavior[1].lower() == 'submissive':
                            labels_curr[data_idx, 5] += counter
                    if len(personality) == 1 and personality[0].lower() == 'neutral':
                        labels_curr[data_idx, 9] += 1.
                    elif len(personality) > 1:
                        counter = 0.
                        if personality[0].lower() == 'extremely':
                            counter = 2.
                        elif personality[0].lower() == 'somewhat':
                            counter = 1.
                        if personality[1].lower() == 'friendly':
                            labels_curr[data_idx, 7] += counter
                        elif personality[1].lower() == 'unfriendly':
                            labels_curr[data_idx, 8] += counter
        labels_curr /= (num_annotators_curr * 2.)
        labels.append(labels_curr)
        num_annotators[didx] = num_annotators_curr
    return np.vstack(labels), num_annotators


def load_edin_data(_path, V, C, joints_to_model, num_labels,
                   frame_drop=1, add_mirrored=False, randomized=True):
    edin_data_files = [os.path.join(_path, 'data_edin_locomotion_train.npz')]
                       # os.path.join(_path, 'data_edin_xsens.npz')]
    if add_mirrored:
        edin_data_dict_file = os.path.join(_path, 'edin_data_dict_with_mirrored_drop_' + str(frame_drop) + '.npz')
    else:
        edin_data_dict_file = os.path.join(_path, 'edin_data_dict_drop_' + str(frame_drop) + '.npz')
    try:
        data_dict = np.load(edin_data_dict_file, allow_pickle=True)['data_dict'].item()
        print('Data file found. Returning data.')
    except FileNotFoundError:
        data_dict = dict()
        data_counter = 0
        num_files = len(edin_data_files)
        num_data = np.zeros(num_files)
        for fidx, edin_data_file in enumerate(edin_data_files):
            file_feature = edin_data_file
            data_loaded = np.load(file_feature)['clips']
            data_sub_sampled = data_loaded[:, ::frame_drop, :]
            num_data[fidx] = len(data_sub_sampled)
            mocap = MocapDataset(V, C, joints_to_model=joints_to_model)
            for idx, data_curr in enumerate(data_sub_sampled):
                data_dict[str(data_counter)] = mocap.get_features_from_data('edin', raw_data=data_curr)
                data_counter += 1
                if add_mirrored:
                    data_dict[str(data_counter)] = mocap.get_features_from_data('edin',
                                                                                raw_data=data_curr,
                                                                                mirrored=True)
                    data_counter += 1
                print('\rData file not found. Processing file {}/{}: {:3.2f}%'.format(
                    fidx + 1, num_files, idx * 100. / num_data[fidx]), end='')
        print('\rData file not found. Processing files: done. Saving...', end='')
        labels, num_annotators = load_edin_labels(_path, np.array(num_data, dtype='int'))
        if add_mirrored:
            labels = np.repeat(labels, 2, axis=0)
        label_partitions = np.append([0], np.cumsum(num_labels))
        for lpidx in range(len(num_labels)):
            labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]] = \
                labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]] / \
                np.linalg.norm(labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]], ord=1, axis=1)[:, None]
        for idx in range(data_counter):
            data_dict[str(idx)]['labels'] = labels[idx]
        np.savez_compressed(edin_data_dict_file, data_dict=data_dict)
        print('done. Returning data.')
    return data_dict, split_data_dict(data_dict, randomized=randomized)


def scale_data(_data, data_max=None, data_min=None):
    _data = _data.astype('float32')
    if data_max is None:
        data_max = np.max(_data)
    if data_min is None:
        data_min = np.min(_data)
    return (_data - data_min) / (data_max - data_min), data_max, data_min


def scale_per_joint(_data, _nframes):
    max_per_joint = np.empty((_data.shape[0], _data.shape[2]))
    min_per_joint = np.empty((_data.shape[0], _data.shape[2]))
    for sidx in range(_data.shape[0]):
        max_per_joint[sidx, :] = np.amax(_data[sidx, :int(_nframes[sidx] - 1), :], axis=0)
        min_per_joint[sidx, :] = np.amin(_data[sidx, :int(_nframes[sidx] - 1), :], axis=0)
    max_per_joint = np.amax(max_per_joint, axis=0)
    min_per_joint = np.amin(min_per_joint, axis=0)
    data_scaled = np.empty_like(_data)
    for sidx in range(_data.shape[0]):
        max_repeated = np.repeat(np.expand_dims(max_per_joint, axis=0), _nframes[sidx] - 1, axis=0)
        min_repeated = np.repeat(np.expand_dims(min_per_joint, axis=0), _nframes[sidx] - 1, axis=0)
        data_scaled[sidx, :int(_nframes[sidx] - 1), :] =\
            np.nan_to_num(np.divide(_data[sidx, :int(_nframes[sidx] - 1), :] - min_repeated,
                                    max_repeated - min_repeated))
    return data_scaled, max_per_joint, min_per_joint


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
