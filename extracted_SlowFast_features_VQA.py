# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from data_loader import VideoDataset_NR_SlowFast_feature

from pytorchvideo.models.hub import slowfast_r50
from torchvision import transforms

#####################################################################################
def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
#fast_pathway 变量被设置为等于输入的 frames。这是用于处理视频帧的其中一个路径。
    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    # slow_pathway 是通过对frames张量进行时间采样而创建的。这涉及使用torch.index_select从原始的frames张量中选择帧。所选择的帧是均匀分布的，采样步长为
    # frames.shape[2] // 4。这实际上将时间分辨率降低了4倍，相对于快速路径来说，慢速路径具有更低的时间分辨率。
    # 这种降采样是双路径模型中的常见操作，以捕捉快速和慢速的时间信息
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway.to(device), fast_pathway.to(device)]
    #frame_list 是一个包含两个张量的列表：slow_pathway 和 fast_pathway。两个张量都使用.to(device)方法移到了指定的device上

    return frame_list
#这个函数为进一步由神经网络模型处理视频数据做准备，通常用于视频动作识别等任务。
# 两个路径用于捕捉不同的时间信息，快速路径关注细节，而慢速路径捕捉更广泛的时间上下文。这是视频分析任务中常见的设计选择，旨在提高性能。
##################################################################################################################

class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])

        self.feature_extraction = torch.nn.Sequential()#用于存储模型的特征提取部分用于存储模型的特征提取部分
        self.slow_avg_pool = torch.nn.Sequential()#用于存储慢速、快速和自适应平均池化层的定义
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0,5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])

        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', slowfast_pretrained_features[5].pool[1])
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)
        
#forward方法定义了前向传播过程。在这里，它首先对输入数据进行特征提取，然后分别通过慢速和快速路径的平均池化层，最后返回慢速和快速特征
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extraction(x)  #特征提取层应用于输入x

            slow_feature = self.slow_avg_pool(x[0])#这行代码将慢速路径的特征提取结果 x[0] 传递给 slow_avg_pool，执行慢速路径的平均池化
            fast_feature = self.fast_avg_pool(x[1])#这行代码将快速路径的特征提取结果 x[1] 传递给 fast_avg_pool，执行快速路径的平均池化

            slow_feature = self.adp_avg_pool(slow_feature)#这行代码将慢速路径的平均池化后的特征再次传递给 adp_avg_pool，执行自适应平均池化
            fast_feature = self.adp_avg_pool(fast_feature)#类似地，这行代码将快速路径的平均池化后的特征再次传递给 adp_avg_pool，执行自适应平均池化
            
        return slow_feature, fast_feature

#训练一个模型并执行特征提取
def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = slowfast()
    

    model = model.to(device)

    resize = config.resize#从配置参数中获取图像调整大小的目标尺寸的代码行
        
    ## training data
    config.database = 'KoNViD-1k'
    if config.database == 'KoNViD-1k':
        videos_dir = 'extract_image2'
        datainfo_test = 'data/KoNViD-1k_data.mat'

        # 使用定义数据转换序列transforms.Compose。这些转换包括transforms.Resize([resize, resize])调整视频帧的大小、resize 是之前设置的目标尺寸
        # transforms.ToTensor()将视频帧转换为张量以及标准化像素值。这些转换通常先应用于输入数据，然后再将其输入神经网络模型
        transformations_test = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])
        #对张量进行像素值标准化，以确保数据位于合适的范围内
    
        trainset = VideoDataset_NR_SlowFast_feature(videos_dir, datainfo_test, transformations_test, resize, 'KoNViD-1k')
#使用上述设置创建了一个名为 trainset 的数据集对象。这个对象是根据 VideoDataset_NR_SlowFast_feature 类创建的，
#它将视频数据的路径、数据信息文件的路径、数据预处理转换、图像大小和数据集名称作为参数传递。这个数据集对象将用于加载和处理数据，以供后续训练和特征提取使用。


    elif config.database == 'youtube_ugc':
        videos_dir = 'youtube_ugc/h264'
        datainfo_test = 'data/youtube_ugc_data.mat'

        transformations_test = transforms.Compose([transforms.Resize([resize, resize]),transforms.ToTensor(),\
            transforms.Normalize(mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225])])
    
        trainset = VideoDataset_NR_SlowFast_feature(videos_dir, datainfo_test, transformations_test, resize, 'youtube_ugc')


    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
        shuffle=False, num_workers=config.num_workers)
#num_workers 是用于数据加载的并行工作线程数量

    # do validation after each epoch
    with torch.no_grad():
        model.eval()  #将模型设置为评估模式,这表示模型将在推断模式下运行，不会记录或更新梯度信息，以减少内存消耗

        #使用enumerate(train_loader)循环遍历训练数据集中的视频
        #一个元组，其中 video 包含视频数据，video_name 包含视频的名称
        for i, (video, video_name) in enumerate(train_loader):
            video_name = video_name[0]  #获取视频的名称，并确保它为字符串类型
            print(video_name)

            #检查是否已经存在特征保存文件夹。如果不存在，则创建该文件夹
            if not os.path.exists(config.feature_save_folder + video_name):
                os.makedirs(config.feature_save_folder + video_name)

            #在接下来的部分，对视频的每一帧进行处理
            for idx, ele in enumerate(video):
                # ele = ele.to(device)
                ele = ele.permute(0, 2, 1, 3, 4)      #对帧数据进行维度置换，这样做可能是为了确保帧数据具有模型预期的正确形状
                inputs = pack_pathway_output(ele, device)
                slow_feature, fast_feature = model(inputs)#模型使用该model(inputs)线路准备和处理视频帧数据，该线路计算慢速和快速特征
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_slow_feature', slow_feature.to('cpu').numpy())
                np.save(config.feature_save_folder + video_name + '/' + 'feature_' + str(idx) + '_fast_feature', fast_feature.to('cpu').numpy())
            # 定义要保存文件的文件夹路径
folder_path = 'KoNViD-1-feature2'
        #
        #     # 确保文件夹存在，如果不存在则创建它
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # file_path = os.path.join(folder_path, np.save)
        # with open(file_path, 'w') as file:
        #     file.write("这是一个示例文件的内容")






        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--database', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--feature_save_folder', type=str, default='')

    config = parser.parse_args()

    main(config)


