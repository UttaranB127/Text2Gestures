# sys
import glob
import numpy as np
import os
import pyttsx3

import utils.constant as constant

from nltk.stem.porter import PorterStemmer
from scipy.io import wavfile
from tqdm import tqdm

from utils.mocap_dataset import MocapDataset


nrc_vad_lexicon_file = '../data/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt'
nrc_vad_lexicon = {}
with open(nrc_vad_lexicon_file, 'r') as nf:
    heading = nf.readline()
    lines = nf.readlines()
    for line in lines:
        line_split = line.split('\t')
        lexeme = line_split[0]
        v = float(line_split[1])
        a = float(line_split[2])
        d = float(line_split[3].split('\n')[0])
        nrc_vad_lexicon[lexeme] = np.array([v, a, d])
porter_stemmer = PorterStemmer()


tts_engine = pyttsx3.init()


def get_vad(lexeme_raw):
    lexeme_lower = lexeme_raw.lower()
    lexeme_stemmed = porter_stemmer.stem(lexeme_lower)
    if lexeme_lower in nrc_vad_lexicon.keys():
        return nrc_vad_lexicon[lexeme_lower]
    if lexeme_stemmed in nrc_vad_lexicon.keys():
        return nrc_vad_lexicon[lexeme_stemmed]
    return np.zeros(3)


def record_and_load_audio(audio_file, text, rate, trimmed=False):
    tts_engine.setProperty('rate', rate)
    tts_engine.save_to_file(text, audio_file)
    tts_engine.runAndWait()
    fs, audio_data = wavfile.read(audio_file)
    audio_data = np.trim_zeros(audio_data)
    if trimmed:
        audio_data = np.trim_zeros(audio_data)
    return fs, audio_data


def get_gesture_splits(sentence, words, num_frames, fps):
    audio_file = 'temp.mp3'
    best_rate = 50
    least_diff = np.inf
    for rate in range(50, 200):
        fs, audio_data = record_and_load_audio(audio_file, sentence, rate, trimmed=True)
        diff = np.abs(len(audio_data) / fs - num_frames / fps)
        if diff < least_diff:
            least_diff = np.copy(diff)
            best_rate = np.copy(rate)
        elif diff > least_diff:
            break
    fs, audio_data = record_and_load_audio(audio_file, sentence, best_rate, trimmed=True)
    sentence_frames = len(audio_data)
    word_frames = []
    fs_s = []
    total_word_frames = 0
    for word in words:
        if len(word) > 0:
            fs, audio_data = record_and_load_audio(audio_file, word, best_rate, trimmed=True)
            fs_s.append(fs)
            word_frames.append(len(audio_data))
            total_word_frames += len(audio_data)
    sampling_ratio = sentence_frames / total_word_frames
    splits = [0]
    for fs, w in zip(fs_s, word_frames):
        splits.append(int(np.ceil(splits[-1] + w * sampling_ratio * fps / fs)))

    if os.path.exists(audio_file):
        os.remove(audio_file)

    return splits


def split_data_dict(data_dict, eval_size=0.1, randomized=True, fill=1):
    num_samples = len(data_dict)
    num_samples_valid = int(round(eval_size * num_samples))
    samples_all = np.array(list(data_dict.keys()), dtype=int)
    if randomized:
        samples_valid = np.random.choice(samples_all, num_samples_valid, replace=False)
    else:
        # samples_valid = samples_all[-num_samples_valid:]
        samples_valid = np.loadtxt('samples_valid.txt').astype(int)
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
        tag_categories = list(np.load(data_dict_file, allow_pickle=True)['tag_categories'])
        max_text_length = np.load(data_dict_file, allow_pickle=True)['max_text_length'].item()
        max_time_steps = np.load(data_dict_file, allow_pickle=True)['max_time_steps'].item()
        print('Data file found. Returning data.')
    except FileNotFoundError:
        data_dict = []
        tag_categories = []
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
                    positions, rotations, base_fps = MocapDataset.load_bvh(bvh_file, channel_map)
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
                        data_dict[tag_data[id]][tag_name] =\
                            tag_data[tag_names.index(tag_name)].replace(' s ', '\'s ').replace(' t ', '\'t ')
                        text_vad = []
                        words = data_dict[tag_data[id]][tag_name].split(' ')
                        for lexeme in words:
                            if lexeme.isalpha():
                                if len(lexeme) == 1 and not (lexeme.lower() is 'a' or lexeme.lower() is 'i'):
                                    continue
                                text_vad.append(get_vad(lexeme))
                        try:
                            data_dict[tag_data[id]][tag_name + ' VAD'] = np.stack(text_vad)
                            data_dict[tag_data[id]]['gesture_splits'] =\
                                get_gesture_splits(data_dict[tag_data[id]][tag_name], words,
                                                   len(data_dict[tag_data[id]]['positions']),
                                                   base_fps / frame_drop)
                        except ValueError:
                            data_dict[tag_data[id]][tag_name + ' VAD'] = np.zeros((0, 3))
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
                    if tag_name is 'Intended emotion' or tag_name is 'Perceived category':
                        data_dict[tag_data[id]][tag_name + ' VAD'] = get_vad(tag_data[tag_names.index(tag_name)])
                print('\rData file not found. Processing file {}/{}: {:3.2f}%'.format(
                    data_counter + 1, num_files, data_counter * 100. / num_files), end='')
            print('\rData file not found. Processing files: done. Saving...', end='')
            np.savez_compressed(data_dict_file,
                                data_dict=data_dict,
                                tag_categories=tag_categories,
                                max_text_length=max_text_length,
                                max_time_steps=max_time_steps)
            print('done. Returning data.')
        elif dataset == 'creative_it':
            mocap_data_dirs = os.listdir(os.path.join(data_path, 'mocap'))
            for mocap_dir in mocap_data_dirs:
                mocap_data_files = glob.glob(os.path.join(data_path, 'mocap/' + mocap_dir + '/*.txt'))
        else:
            raise FileNotFoundError('Dataset not found.')

    return data_dict, tag_categories, max_text_length, max_time_steps


def build_vocab_idx(word_instants, min_word_count):
    # word to index dictionary
    word2idx = {
        constant.BOS_WORD: constant.BOS,
        constant.EOS_WORD: constant.EOS,
        constant.PAD_WORD: constant.PAD,
        constant.UNK_WORD: constant.UNK,
    }

    full_vocab = set(w for sent in word_instants for w in sent)
    print('Original Vocabulary size: {}'.format(len(full_vocab)))

    word_count = {w: 0 for w in full_vocab}

    # count word frequency in the given dataset
    for sent in word_instants:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)  # add word to dictionary with index
            else:
                ignored_word_count += 1

    print('Trimmed vocabulary size: {}\n\teach with minimum occurrence: {}'.format(len(word2idx), min_word_count))
    print('Ignored word count: {}'.format(ignored_word_count))

    return word2idx


def build_embedding_table(embedding_path, target_vocab):
    def load_emb_file(_embedding_path):
        vectors = []
        idx = 0
        _word2idx = dict()
        _idx2word = dict()
        with open(_embedding_path, 'r') as f:
            for l in tqdm(f):
                line = l.split()
                word = line[0]
                w_vec = np.array(line[1:]).astype(np.float)

                vectors.append(w_vec)
                _word2idx[word] = idx
                _idx2word[idx] = word
                idx += 1

        return np.array(vectors), _word2idx, _idx2word

    vectors, word2idx, idx2word = load_emb_file(embedding_path)
    dim = vectors.shape[1]

    embedding_table = np.zeros((len(target_vocab), dim))
    for k, v in target_vocab.items():
        try:
            embedding_table[v] = vectors[word2idx[k]]
        except KeyError:
            embedding_table[v] = np.random.normal(scale=0.6, size=(dim,))

    return embedding_table


def load_data_with_glove(_path, dataset, embedding_src, frame_drop=1, add_mirrored=False):
    data_path = os.path.join(_path, dataset)
    data_dict_file = os.path.join(data_path, 'data_dict_glove_drop_' + str(frame_drop) + '.npz')
    try:
        data_dict = np.load(data_dict_file, allow_pickle=True)['data_dict'].item()
        word2idx = np.load(data_dict_file, allow_pickle=True)['word2idx'].item()
        embedding_table = np.load(data_dict_file, allow_pickle=True)['embedding_table']
        tag_categories = list(np.load(data_dict_file, allow_pickle=True)['tag_categories'])
        max_time_steps = np.load(data_dict_file, allow_pickle=True)['max_time_steps'].item()
        print('Data file found. Returning data.')
    except FileNotFoundError:
        data_dict = []
        word2idx = []
        embedding_table = []
        tag_categories = []
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

            all_texts = [[]] * len(tag_files)
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
                        all_texts[data_counter] = [e for e in str.split(tag_data[tag_names.index(tag_name)]) if
                                                   e.isalnum()]
                        data_dict[tag_data[id]][tag_name] = tag_data[tag_names.index(tag_name)]
                        text_length = len(data_dict[tag_data[id]][tag_name])
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
                print('\rData file not found. Reading data files {}/{}: {:3.2f}%'.format(
                    data_counter + 1, num_files, data_counter * 100. / num_files), end='')
            print('\rData file not found. Reading files: done.')
            print('Preparing embedding table:')
            word2idx = build_vocab_idx(all_texts, min_word_count=0)
            embedding_table = build_embedding_table(embedding_src, word2idx)
            np.savez_compressed(data_dict_file,
                                data_dict=data_dict,
                                word2idx=word2idx,
                                embedding_table=embedding_table,
                                tag_categories=tag_categories,
                                max_time_steps=max_time_steps)
            print('done. Returning data.')
        elif dataset == 'creative_it':
            mocap_data_dirs = os.listdir(os.path.join(data_path, 'mocap'))
            for mocap_dir in mocap_data_dirs:
                mocap_data_files = glob.glob(os.path.join(data_path, 'mocap/' + mocap_dir + '/*.txt'))
        else:
            raise FileNotFoundError('Dataset not found.')

    return data_dict, word2idx, embedding_table, tag_categories, max_time_steps
