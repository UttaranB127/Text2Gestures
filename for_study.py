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
print('narration')
for key in list(data_dict_eval.keys()):
    if tag_categories[4][np.where(data_dict_eval[key]['Acting task'])[0][0]] == 'Narration':
        print('{}\t{}\t\t{}\t{}'.format(key,
                                      tag_categories[0][np.where(data_dict_eval[key]['Intended emotion'])[0][0]],
                                      tag_categories[2][np.where(data_dict_eval[key]['Perceived category'])[0][0]],
                                      data_dict_eval[key]['Text']))
print('\n\nConversation')
for key in list(data_dict_eval.keys()):
    if tag_categories[4][np.where(data_dict_eval[key]['Acting task'])[0][0]] == 'Sentence':
        print('{}\t{}\t{}\t{}'.format(key,
                                      tag_categories[0][np.where(data_dict_eval[key]['Intended emotion'])[0][0]],
                                      tag_categories[2][np.where(data_dict_eval[key]['Perceived category'])[0][0]],
                                      data_dict_eval[key]['Text']))

print('\n\nOthers')
for key in list(data_dict_eval.keys()):
    if tag_categories[4][np.where(data_dict_eval[key]['Acting task'])[0][0]] == 'Nonverbal':
        print('{}\t{}\t{}\t{}'.format(key,
                                      tag_categories[0][np.where(data_dict_eval[key]['Intended emotion'])[0][0]],
                                      tag_categories[2][np.where(data_dict_eval[key]['Perceived category'])[0][0]],
                                      data_dict_eval[key]['Text']))

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
