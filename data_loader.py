import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2

class VideoDataset_images_with_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    # data_dir存储视频数据(图像或帧)的目录,data_dir_3D:可以存储3d运动特性的目录,filename_path:与数据集关联的MAT文件(可能包含元数据)的路径,
    # transform: 适用于视频帧的转换功能,database_name:显示数据集名称的字符串,crop_size:视频帧的尺寸,feature_type一个表示所使用的运动特性类型的字符串。
    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, crop_size, feature_type):
        super(VideoDataset_images_with_motion_features, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
                #根据MAT文件中的信息，将视频名称和评分信息存储在类的属性
            self.video_names = video_names
            self.score = score
            # print("已经完成训练集的加载")

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][0]
            for i in index_all:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score

        elif database_name == 'LSVQ_train':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            # 它指定CSV文件中列的名称。这些列名称对应于数据的各种属性
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            # (pd.read_csv)附有指定的列名称。…header = 0参数表明CSV文件的第一行包含列名称。…sep = ','论点指定列用逗号分隔, names = column_names
            # 将提供的列名称分配给数据。…index_col = False争论妨碍阅读索引列, 以及encoding = "utf-8-sig"指定字符编码
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()
            # 它从加载的数据中提取"名称"列和"MOS"列, 并将它们转换为列表。这些名单分配给self.video_names和self.score分别使它们可以作为此类实例的属性访问

        elif database_name == 'LSVQ_test':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        elif database_name == 'LSVQ_test_1080p':
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
            dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            self.video_names = dataInfo['name'].tolist()
            self.score = dataInfo['mos'].tolist()

        self.crop_size = crop_size   #代表了在数据预处理过程中视频帧被剪掉的大小
        self.videos_dir = data_dir   #存储视频数据(图像或帧)的目录
        self.data_dir_3D = data_dir_3D   #这是一个可以存储3d运动特性的目录。这意味着数据集包括视频框架和额外的3d运动功能,可以在你的深度学习模型中使用
        self.transform = transform   #此函数表示在模型中使用之前将应用于视频帧的数据预处理转换,这个函数通常包括图像调整大小、标准化和其他必要的操作
        self.length = len(self.video_names)  #它表示数据集中的视频总数
        self.feature_type = feature_type   #指示正在使用的运动特性的类型,这是创建此类实例时指定的
        self.database_name = database_name

    def __len__(self):
        return self.length

#它用于返回数据集中的单个样本
    def __getitem__(self, idx):
        if self.database_name == 'KoNViD-1k' \
            or self.database_name == 'youtube_ugc' :
            video_name = self.video_names[idx]
            #去掉文件扩展名 '.mp4'（如果存在的话）来生成 video_name_str。
            #这是因为在一些数据集中，视频文件名通常以 '.mp4' 结尾，但在处理视频时不需要这个扩展名，只需要基本的视频名称
            video_name_str = video_name[:-4]
        elif self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_name = self.video_names[idx] + '.mp4'
            video_name_str = video_name[:-4]

        video_score = torch.FloatTensor(np.array(float(self.score[idx])))  #把录像的比分转换成torch.floatTensor

#是视频帧所在的目录的完整路径，通过将 self.videos_dir 和 video_name_str 连接而成
        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3  #视频通道数目

        video_height_crop = self.crop_size   #剪裁视频帧的高度及宽度
        video_width_crop = self.crop_size
       
        if self.database_name == 'KoNViD-1k' or self.database_name == 'LSVQ_train' or self.database_name == 'LSVQ_test' or self.database_name == 'LSVQ_test_1080p':
            video_length_read = 8     #要阅读的帧数目
        elif self.database_name == 'youtube_ugc':
            video_length_read = 20

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])
        #初始化一个空张量transformed_video存储预处理的视频帧
        #这个张量的形状为 [video_length_read, video_channel, video_height_crop, video_width_crop]，表示视频的时间维度、通道维度以及剪裁后的空间维度

#这个循循环将处理视频中的每一帧，将它们依次存储到 transformed_video 张量中，为后续处理和特征提取做准备
        for i in range(video_length_read):   #然后,它进入一个循环,遍历视频的每一帧
            imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')#方法将图像转换为 RGB 模式，以确保图像具有三个通道
            read_frame = self.transform(read_frame)#通过调用 self.transform(read_frame)，对图像进行数据预处理的转换，这通常包括将图像缩放到模型期望的尺寸，转换为张量，并标准化像素值
            transformed_video[i] = read_frame

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])#初始化一个空的 transformed_feature 张量：这个张量的形状是[video_length_read, 2048]表示要读取的帧数
            for i in range(video_length_read):
                i_index = i   
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)#是在将3D运动特性加载到PyTorch张量中的一种常见操作，以便将其集成到神经网络模型中
                feature_3D = feature_3D.squeeze()#是 PyTorch 中的一个函数，它用于删除张量中尺寸为1的维度，以简化张量的形状
                transformed_feature[i] = feature_3D#将处理后的慢速运动特性存储在 transformed_feature 张量的第 i 个位置
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
                # 它加载和处理快速的运动特性
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048+256])
            # 它为每一个框架加载缓慢和快速的运动特性, 并将它们连接成一个单一的特性张量
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(os.path.join(feature_folder_name,  'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name,  'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

       
        return transformed_video, transformed_feature, video_score, video_name

    
 







class VideoDataset_images_VQA_dataset_with_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, data_dir_3D ,filename_path, transform, database_name, crop_size, feature_type, exp_id, state = 'train'):
        super(VideoDataset_images_VQA_dataset_with_motion_features, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][exp_id]
            #选择在训练集或验证集中使用的索引。如果 state 为 'train'，则选择前 80% 的索引，如果 state 为 'val'，则选择后 20% 的索引
            if state == 'train':
                index = index_all[:int(n*0.8)]
            elif state == 'val':
                index = index_all[int(n*0.8):]

#遍历视频列表并将视频名称和对应的得分添加到 video_names 和 score 列表中
            for i in index:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            self.video_names = video_names
            self.score = score

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][exp_id]
            if state == 'train':
                index = index_all[:int(n*0.8)]
            elif state == 'val':
                index = index_all[int(n*0.8):]

            for i in index:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score


        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3#视频帧的通道数

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size
       
        if self.database_name == 'KoNViD-1k':
            video_length_read = 8
        elif self.database_name == 'youtube_ugc':
            video_length_read = 10

#video_length_read：这个变量表示要从视频中读取的帧数
        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])             


        for i in range(video_length_read):
            if self.database_name == 'youtube_ugc':
                imge_name = os.path.join(path_name, '{:03d}'.format(i*2) + '.png')
            else:
                imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i/2)   
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i/2) 
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048+256])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i/2) 
                feature_3D_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

       
        return transformed_video, transformed_feature, video_score, video_name


class VideoDataset_NR_LSVQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize, is_test_1080p = False):
        super(VideoDataset_NR_LSVQ_SlowFast_feature, self).__init__()
        if is_test_1080p:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
        else:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1',\
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1',\
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']
                                        
        dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.score = dataInfo['mos']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx])))/20

        filename=os.path.join(self.videos_dir, video_name + '.mp4')

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap=cv2.VideoCapture(filename)

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        video_clip = int(video_length/video_frame_rate)
       
        video_clip_min = 8

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []
        
        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1


        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 
        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_score, video_name



class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, data_dir, filename_path, transform, resize, database_name):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            for i in range(n):
                video_names.append(dataInfo['video_names'][i][0][0])
            self.video_names = video_names

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            for i in range(n):
                video_names.append(dataInfo['video_names'][i][0][0])
            self.video_names = video_names

        self.transform = transform           
        self.videos_dir = data_dir
        self.resize = resize
        self.database_name = database_name
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]#根据给定索引 idx 获取对应的视频名称
        video_name_str = video_name[:-4]#从视频名称中去掉扩展名 ".mp4"，以获得视频的名称字符串
        filename=os.path.join(self.videos_dir, video_name)#构建视频文件的完整路径，这包括了视频存储目录 self.videos_dir 和视频名称

        video_capture = cv2.VideoCapture()#创建一个 OpenCV 视频捕获对象
        video_capture.open(filename)#打开视频文件以准备读取视频帧
        cap=cv2.VideoCapture(filename)#创建另一个 OpenCV 视频捕获对象 cap，用于获取视频的信息，如帧总数和帧速率

        video_channel = 3
        
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#取视频的总帧数，使用 cap.get 方法访问 cv2.CAP_PROP_FRAME_COUNT 属性
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))#获取视频的帧速率，使用 cap.get 方法访问 cv2.CAP_PROP_FPS 属性，将其四舍五入为整数

# 检查视频的帧速率是否为0，如果是0，意味着无法获取有效的帧速率信息，通常是因为视频文件损坏或不正确。
# 在这种情况下，将 video_clip 设置为 10，这是一个默认值，以便后续处理不会失败
        if video_frame_rate == 0:
           video_clip = 10
# 如果视频帧速率不为0，那么计算 video_clip，即期望处理的视频剪辑数目。
# 这是通过将视频总帧数 video_length 除以帧速率 video_frame_rate 得到的，以确保每个剪辑的时间长度相对均匀
        else:
            video_clip = int(video_length/video_frame_rate)

        if self.database_name == 'KoNViD-1k':
            video_clip_min = 8
        elif self.database_name == 'youtube_ugc':
            video_clip_min = 20

# 定义用于每个视频剪辑的帧数，无论视频的实际帧速率如何，每个剪辑都包含 32 帧
        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

# 创建一个空列表 transformed_video_all，用于存储每个视频剪辑的帧
        transformed_video_all = []
        # 初始化 video_read_index，以跟踪已读取的视频帧数量，初始值为0
        video_read_index = 0
        for i in range(video_length):
            # has_frames 是一个布尔值，如果成功读取了一帧，则为 True，否则为 False。frame 包含了读取的视频帧的图像数据
            has_frames, frame = video_capture.read()
            if has_frames:
                # 将读取的视频帧数据转换为 PIL 图像对象，并确保颜色通道的顺序为 RGB
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 使用定义的数据预处理 transform 来处理当前的视频帧。这通常包括缩放、剪裁、标准化等操作
                read_frame = self.transform(read_frame)
                # 将处理后的视频帧存储在名为 transformed_frame_all 的张量中，以 video_read_index 作为索引。
                # 然后，video_read_index 递增，以跟踪已读取的帧的数量
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

# 确保已经读取的帧数小于视频的总帧数
        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                # 在该循环中，使用video_read_index - 1的索引来复制之前已经读取的最后一帧。
                # 这是为了填充剩余的未读取帧的内容，以确保transformed_frame_all的所有帧都被填满。这种填充策略可以保持一致的视频帧数量，以便后续的处理和分析
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]
 
        video_capture.release()
# video_clip表示要创建的视频剪辑数量
        for i in range(video_clip):
            # 在循环内部，首先创建一个名为transformed_video的空张量，
            # 其形状为[video_length_clip, video_channel, self.resize, self.resize]，表示要创建的视频剪辑的维度
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            # 它检查当前视频剪辑是否可以从视频的当前位置（i*video_frame_rate）提取，并且能够包含video_length_clip帧。
            # 这是通过检查(i*video_frame_rate + video_length_clip) <= video_length来实现的。
            # 如果是这样，它就提取这个范围内的帧，从i*video_frame_rate到(i*video_frame_rate + video_length_clip)
            if (i*video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[i*video_frame_rate : (i*video_frame_rate + video_length_clip)]
            else:
            # 使用一个内部循环for j in range((video_length - i*video_frame_rate), video_length_clip)，来填充其余的帧。
            # 这些帧被复制自最后一帧，以保持视频剪辑的长度一致
                transformed_video[:(video_length - i*video_frame_rate)] = transformed_frame_all[i*video_frame_rate :]
                for j in range((video_length - i*video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i*video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        # 这段代码的目的是确保至少创建 video_clip_min 个视频剪辑。
        # 如果创建的视频剪辑数量小于这个最小数量，它会复制最后一个已创建的视频剪辑，以填补不足的部分，以便满足最小数量的要求。
        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])
       
        return transformed_video_all, video_name_str
