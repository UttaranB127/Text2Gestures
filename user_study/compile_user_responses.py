import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from scipy import stats
from sklearn.cluster import KMeans

sns.set(color_codes=True, font='cmss10')
sns.set_context('paper', font_scale=2.)


def get_emo_as_adjective(emo):
    emo_as_adjective = ''
    if emo == 'neutral':
        emo_as_adjective = 'Neutral'
    elif emo == 'joy':
        emo_as_adjective = 'Joyous'
    elif emo == 'shame':
        emo_as_adjective = 'Ashamed'
    elif emo == 'amusement':
        emo_as_adjective = 'Amused'
    elif emo == 'pride':
        emo_as_adjective = 'Proud'
    elif emo == 'sadness':
        emo_as_adjective = 'Sad'
    elif emo == 'surprise':
        emo_as_adjective = 'Surprised'
    elif emo == 'anger':
        emo_as_adjective = 'Angry'
    elif emo == 'fear':
        emo_as_adjective = 'Afraid'
    elif emo == 'relief':
        emo_as_adjective = 'Relieved'
    elif emo == 'disgust':
        emo_as_adjective = 'Disgusted'
    return emo_as_adjective


def append_unique(the_list, the_item):
    the_list.append(the_item)
    return list(set(the_list))


def get_nearby_emos(emos_list):
    if 'Afraid' in emos_list or 'Ashamed' in emos_list:
        emos_list = append_unique(emos_list, 'Afraid')
        emos_list = append_unique(emos_list, 'Ashamed')
        emos_list = append_unique(emos_list, 'Neutral')
    if 'Joyous' in emos_list or 'Surprised' in emos_list or 'Amused' in emos_list:
        emos_list = append_unique(emos_list, 'Joyous')
        emos_list = append_unique(emos_list, 'Surprised')
        emos_list = append_unique(emos_list, 'Amused')
        emos_list = append_unique(emos_list, 'Neutral')
    if 'Angry' in emos_list or 'Disgusted' in emos_list or 'Proud' in emos_list:
        emos_list = append_unique(emos_list, 'Angry')
        emos_list = append_unique(emos_list, 'Disgusted')
        emos_list = append_unique(emos_list, 'Neutral')
    if 'Sad' in emos_list or 'Relieved' in emos_list:
        emos_list = append_unique(emos_list, 'Sad')
        emos_list = append_unique(emos_list, 'Relieved')
        emos_list = append_unique(emos_list, 'Amused')
    return emos_list


with open('data_dict_eval.pkl', 'rb') as ef:
    data_dict_eval = pickle.load(ef)
with open('tag_categories.pkl', 'rb') as tf:
    tag_categories = pickle.load(tf)

data_study = pd.read_csv('Text2Gestures_August+30,+2020_22.19.csv').loc[2:]\
    .dropna(subset=['urlList', 'gt_url', 'S1', 'S2', 'S3', 'S4'])
ax = sns.catplot(x='S0.1', hue='S0.2', data=data_study, kind='count', palette='colorblind')
ax.set(xlabel='Sex')
ax._legend.set_title('Age Group')
# ax.savefig('plots/demographic.png')

video_order_s2 = ['gt', 'test', 'gt', 'gt', 'test', 'test']

test_videos = data_study['urlList'].to_list()
gt_videos = data_study['gt_url'].to_list()
test_videos_s1 = []
test_videos_s2 = []
test_videos_s3 = []
safe_list = []
for t_idx, test_video in enumerate(test_videos):
    video_nums = test_video.split(',')
    if len(video_nums) == 15:
        safe_list.append(t_idx)
        test_videos_s1.append(video_nums[:6])
        test_videos_s2.append(video_nums[6:9])
        test_videos_s3.append(video_nums[9:])

responses_c1 = [el for el_idx, el in enumerate(data_study['S1'].to_list()) if el_idx in safe_list]
responses_c2 = [el for el_idx, el in enumerate(data_study['S2'].to_list()) if el_idx in safe_list]
responses_c3 = [el for el_idx, el in enumerate(data_study['S3'].to_list()) if el_idx in safe_list]
responses_c4 = [el for el_idx, el in enumerate(data_study['S4'].to_list()) if el_idx in safe_list]

responses_naturalness = []
responses_acting_task = []
for i in range(1, 7):
    responses_naturalness.append(data_study['S2.' + str(i) + '_1'].to_list())
    responses_acting_task.append(data_study['S3.' + str(i)].to_list())

emo_total = 0
emo_matched = 0
for all_v_idx, all_r1, all_r2, all_r3, all_r4 in zip(test_videos_s1,
                                                     responses_c1, responses_c2,
                                                     responses_c3, responses_c4):
    all_r1 = all_r1.split(',')
    all_r2 = all_r2.split(',')
    all_r3 = all_r3.split(',')
    all_r4 = all_r4.split(',')
    for v_idx, r1, r2, r3, r4 in zip(all_v_idx, all_r1, all_r2, all_r3, all_r4):
        emo_total += 1
        key = str(int(v_idx)).zfill(6)
        gt_emo = get_nearby_emos(
            [get_emo_as_adjective(tag_categories[0]
                                  [np.where(data_dict_eval[key]['Intended emotion'])[0][0]]),
             get_emo_as_adjective(tag_categories[2]
                                  [np.where(data_dict_eval[key]['Perceived category'])[0][0]])
             ])
        if r1.strip() in gt_emo or r2.strip() in gt_emo or r3.strip() in gt_emo or r4.strip() in gt_emo:
            emo_matched += 1

naturalness_total = {'gt': np.zeros(5), 'test': np.zeros(5)}

for responses_per_sample, sample_source in zip(responses_naturalness, video_order_s2):
    for response_per_user in responses_per_sample:
        if isinstance(response_per_user, str):
            if 'great' in response_per_user.lower():
                naturalness_total[sample_source][4] += 1
            if 'good' in response_per_user.lower():
                naturalness_total[sample_source][3] += 1
            if 'ok' in response_per_user.lower():
                naturalness_total[sample_source][2] += 1
            if 'realistic' in response_per_user.lower():
                naturalness_total[sample_source][1] += 1
            if 'unnatural' in response_per_user.lower():
                naturalness_total[sample_source][1] += 1

acting_task_total = 0
acting_task_matched = 0
for s_idx, responses_per_sample in enumerate(responses_acting_task):
    responses_per_sample = [el for el_idx, el in enumerate(responses_per_sample) if el_idx in safe_list]
    for v_idx, response_per_user in enumerate(responses_per_sample):
        acting_task_total += 1
        key = str(int(test_videos_s3[v_idx][s_idx])).zfill(6)
        acting_task = tag_categories[4][np.where(data_dict_eval[key]['Acting task'])[0][0]]
        if 'sen' in acting_task.lower() and 'conv' in response_per_user:
            acting_task_matched += 1
        elif not ('sen' in acting_task.lower() or 'conv' in response_per_user):
            acting_task_matched += 1

temp = 1
