import numpy as np
import os
import pandas as pd
import cv2
import scipy.io as scio


def extract_key_frames(videos_dir, video_name, save_folder, max_frames=16, diff_threshold=30):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    #打开视频文件
    video_capture.open(filename)

    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))

    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the height of frames
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames
    #获取视频的相关信息，包括帧数、帧率、帧的高度和宽度。

    if video_height > video_width:
        video_width_resize = 520
        video_height_resize = int(video_width_resize / video_width * video_height)
    else:
        video_height_resize = 520
        video_width_resize = int(video_height_resize / video_height * video_width)
        #根据视频帧的高度和宽度，调整图像大小，以适应后续处理

    dim = (video_width_resize, video_height_resize)

    # 计算间隔以确保不超过16帧
    frame_interval = max(1, video_length // max_frames)

    ret, prev_frame = video_capture.read()
    prev_frame = cv2.resize(prev_frame, dim) if ret else None

    frame_idx, key_frame_count = 0, 0

    while ret:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, dim)

        if frame_idx % frame_interval == 0:
            # 计算当前帧和前一帧之间的差异
            diff = cv2.absdiff(frame, prev_frame)
            non_zero_count = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))

            if non_zero_count > diff_threshold:
                # 保存关键帧
                exit_folder(os.path.join(save_folder, video_name_str))
                # 使用 key_frame_count 作为文件名的序号
                cv2.imwrite(os.path.join(save_folder, video_name_str, '{:03d}.png'.format(key_frame_count)), frame)
                key_frame_count += 1

                if key_frame_count >= max_frames:
                    break

        prev_frame = frame
        frame_idx += 1

    video_capture.release()
    print('Extracted', key_frame_count, 'key frames from', video_name)



def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap = cv2.VideoCapture(filename)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # the width of frames

    if video_height > video_width:
        video_width_resize = 520
        video_height_resize = int(video_width_resize / video_width * video_height)
    else:
        video_height_resize = 520
        video_width_resize = int(video_height_resize / video_height * video_width)

    dim = (video_width_resize, video_height_resize)

    video_read_index = 0

    frame_idx = 0

    video_length_min = 8

    for i in range(video_length):
        has_frames, frame = video_capture.read()
        if has_frames:
            # key frame
            if (video_read_index < video_length) and (frame_idx % video_frame_rate == 0):
                read_frame = cv2.resize(frame, dim)
                exit_folder(os.path.join(save_folder, video_name_str))
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                         '{:03d}'.format(video_read_index) + '.png'), read_frame)
                video_read_index += 1
            frame_idx += 1

    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                     '{:03d}'.format(i) + '.png'), read_frame)

    return


def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return


videos_dir = 'videos/KoNViD_1k'
filename_path = 'data/KoNViD-1k_data.mat'

dataInfo = scio.loadmat(filename_path)
n_video = len(dataInfo['video_names'])
video_names = []

for i in range(n_video):
    video_names.append(dataInfo['video_names'][i][0][0])

save_folder = 'extract_image2'
for i in range(n_video):
    video_name = video_names[i]
    print('start extract {}th video: {}'.format(i, video_name))
    extract_key_frames(videos_dir, video_name, save_folder)
