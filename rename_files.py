import os

# video_categories = ['ours', 'no_aff', 'no_pose', 'no_ang']
video_categories = ['2_7_39']
base_path = 'W:/Gamma/Gestures'
videos_file_path = os.path.join(base_path, 'videos')

for cat in video_categories:
    for seq_num in range(1, 5):
        seq = os.path.join(videos_file_path, 'individual_frames_' + cat + '/seq' + str(seq_num))
        print('{}'.format(seq), end='')
        files = os.listdir(seq)
        for fidx, file in enumerate(files):
            os.rename(os.path.join(seq, file), os.path.join(seq, str(fidx).zfill(6) + '.png'))
        print('\tdone.')
