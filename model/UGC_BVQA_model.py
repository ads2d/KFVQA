import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.BaseModel import SelfAttention1D
from model.ECAAttention import ECAAttention1D
import numpy as np
#from thop import profile
# from sklearn import svm


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


##in_planes：输入通道数,out_planes：输出通道数,stride：卷积步幅（默认为1）,groups：用于分组卷积的组数（默认为1）,dilation：卷积膨胀率（默认为1）
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    ##它返回一个具有指定配置的nn.Conv2d层。
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1      #expansion 类属性设置为 1，表示该基本块的通道数不会发生扩展
    __constants__ = ['downsample']    #定义不需要序列化的成员变量
#downsample：可选的下采样层，用于调整输入特征图的大小,base_width：基本宽度（默认为 64）,dilation：卷积膨胀率（默认为 1）,norm_layer：用于标准化的层（默认为 nn.BatchNorm2d）
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        #__init__ 方法是类的构造函数，用于初始化 BasicBlock 实例
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        #创建了两个卷积层 self.conv1 和 self.conv2，以及相应的批标准化层 self.bn1 和 self.bn2。这些层被用于构建块内的卷积和标准化操作
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):    #forward 方法定义了数据在基本块内的前向传播过程
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#这段代码定义了一个深度残差网络（ResNet）中的瓶颈块（Bottleneck）
class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
#block：ResNet 中使用的基本块类型，可以是 BasicBlock 或 Bottleneck
#layers：一个包含四个整数的列表，表示每个阶段中的块数
#num_classes：分类任务的类别数（默认为 1000，适用于 ImageNet 数据集）
#zero_init_residual：如果为 True，则将残差连接的初始权重初始化为零（默认为 False）
#groups：卷积分组数（默认为 1）width_per_group：每个分组中的通道数（默认为 64）
#replace_stride_with_dilation：控制是否使用膨胀卷积的参数，应该是一个包含三个布尔值的列表或 None
#norm_layer：用于标准化的层（默认为 nn.BatchNorm2d
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #每个阶段包含多个块，块的数量由 layers 参数控制。
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#是一个自适应平均池化层，将最后的特征图池化成大小为 (1, 1) 的张量

        # three stage spatial features (avg + std) + motion
        self.quality = self.quality_regression(4096+2048+1024+2048+256, 128,1)


        #是一个质量回归层，它接收多个阶段的特征图作为输入（通过连接它们），并输出质量相关的信息。

#在最后的代码块中，模型的参数进行了初始化。卷积层使用 Kaiming 初始化，批标准化和分组标准化层的权重被初始化为1，偏差初始化为0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        #这个初始化操作用于帮助模型更快地收敛
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

# _make_layer 函数用于创建 ResNet 中的一个阶段（stage），这个阶段包含多个重复的块（blocks）。
# 参数包括：
# block：要使用的基本块类型，可以是 BasicBlock 或 Bottleneck。
# planes：输出通道数。
# blocks：阶段中要重复的块的数量。
# stride：卷积步幅（默认为 1）。
# dilate：一个布尔值，控制是否使用膨胀卷积（默认为 False）
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            #如果需要下采样（stride 不等于 1或输入通道数不等于输出通道数的情况），会创建一个下采样层 downsample，用于调整输入特征图的大小
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
#接着，它根据 blocks 参数的值重复添加相应数量的块到 layers 列表中，这些块是通过调用 block 构造函数创建的，包括了块内的卷积、标准化等操作
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
#最后，函数返回一个包含所有块的 nn.Sequential 容器。
        return nn.Sequential(*layers)

#quality_regression 函数定义了一个简单的质量回归模块，在函数内部，它创建了一个包含两个线性层的顺序模块，这些层用于进行质量回归预测
    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block


    # def quality_regression(self, in_channels, middle_channels, out_channels):
    #     # 创建SVM回归模型
    #       model = svm.SVR(kernel='linear')
    #
    #       return model



    #_forward_impl 方法定义了 ResNet 模型的前向传播逻辑
    def _forward_impl(self, x, x_3D_features):
        # print("进入_forward_impl")
        # See note [TorchScript super()]
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x_3D: batch x frames x (2048 + 256)
        x_3D_features_size = x_3D_features.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        # x_3D: batch * frames x (2048 + 256)
        x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])
        # 执行了特征提取的初始操作，通常包括卷积、批量归一化、激活函数和池化。这有助于提取输入 x 中的低级特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print("303")

        # 这些部分是深度卷积神经网络的主体。它们包含多个卷积层，通常分为多个阶段。
        # 在这个过程中，特征被多次抽取和变换，以获得更高级的抽象特征。这有助于模型理解输入数据中的复杂模式
        x = self.layer1(x)
        x = self.layer2(x)
        x_avg2 = self.avgpool(x)
        x_std2 = global_std_pool2d(x)
        # print("x_avg2.shape:", x_avg2.shape)
        # print("x_std2.shape:", x_std2.shape)
        # print("x",x.shape)

        x = self.layer3(x)
        x_avg3 = self.avgpool(x)
        x_std3 = global_std_pool2d(x)
        # print("x_avg3.shape:", x_avg3.shape)
        # print("x_std3.shape:", x_std3.shape)
        # print("x", x.shape)

        x = self.layer4(x)
        x_avg4 = self.avgpool(x)
        x_std4 = global_std_pool2d(x)
        # print("x_avg4.shape:", x_avg4.shape)
        # print("x_std4.shape:", x_std4.shape)
        # print("x", x.shape)
        # print("----------------")
        # print("即将执行torch.cat")



        # 每个阶段的末尾，都有池化操作（avgpool）和标准差池化操作（global_std_pool2d）。
        # 这些池化操作有助于降低数据的维度，同时保留重要信息。x_avg2、x_std2、x_avg3、x_std3、x_avg4、x_std4 分别代表不同阶段的平均池化结果和标准差池化结果
        x = torch.cat((x_avg2, x_std2, x_avg3, x_std3, x_avg4, x_std4), dim = 1)
        # print("before:", x.shape)



        # x: batch * frames x (2048*2 + 1024*2 + 512*2)
        # x = torch.flatten(x, 1)
        # x = torch.cat((x, x_3D_features), dim=1)
        # x: batch * frames x (2048*2 + 1024*2 + 512*2 + 2048 + 512)

        fc = nn.Sequential(nn.Linear(x.shape[1], 1024),
                           nn.Linear(1024, 6),
                           nn.Sigmoid())
        fc.cuda()

        weight = fc(x)
        # print("weight.shape:", weight.shape)
        x_avg2 = weight[1][0] * x_avg2
        x_std2 = weight[1][1] * x_std2
        x_avg3 = weight[1][2] * x_avg3
        x_std3 = weight[1][3] * x_std3
        x_avg4 = weight[1][4] * x_avg4
        x_std4 = weight[1][5] * x_std4

        x = torch.cat((x_avg2, x_std2, x_avg3, x_std3, x_avg4, x_std4), dim=1)
        # print("after shape:", x.shape)
        x = torch.flatten(x, 1)

        #将连接的特征与 x_3D_features 连接，输入到质量回归模块 self.quality 中，以进行质量预测
        x_before = torch.cat((x, x_3D_features), dim = 1)
        att = SelfAttention1D(x_before.shape[1])
        att.cuda()
        x_m = att(x_before)
        x = x_before + x_m




        # x: batch * frames x 1
        x = self.quality(x)
        # x: batch x frames
        x = x.view(x_size[0],x_size[1])

        # x: batch x 1
        x = torch.mean(x, dim = 1)

        return x

    def forward(self, x, x_3D_features):
        return self._forward_impl(x, x_3D_features)

        # 创建原始模型并加载预训练权重（如果需要）
        # original_model = YourOriginalModel()
        # 如果有预训练权重，可以加载它们
        # original_model.load_state_dict(torch.load('your_pretrained_model.pth'))
        # 创建 QualityAwareModel，选择主要块的数量
        # num_main_blocks = 10  # 选择前 10 个块作为主要块
        # quality_aware_model = QualityAwareModel(original_model, num_main_blocks)

# _resnet 函数用于创建 ResNet 模型。
# 参数包括：
# arch：模型架构的名称。
# block：要使用的基本块类型，可以是 BasicBlock 或 Bottleneck。
# layers：一个包含四个整数的列表，表示每个阶段中的块数。
# pretrained：是否加载预训练的权重。
# progress：是否显示下载进度。
# **kwargs：其他模型构建参数。
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet34'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input, ))
    # print('The flops is {:.4f}, and the params is {:.4f}'.format(flops/10e9, params/10e6))
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet50'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
    #                **kwargs)
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet101'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
    #                **kwargs)
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet152'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    #return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
       #            pretrained, progress, **kwargs)
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnext50_32x4d'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    # return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
    #                pretrained, progress, **kwargs)
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnext101_32x8d'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)




if __name__ == "__main__":

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = resnet50(pretrained=False).to(device)
    # print(model)
    from thop import profile
    from thop import clever_format

    input = torch.randn(8,8,3,448,448)
    input_3D = torch.randn(8, 8, 2048+256)
    flops, params = profile(model, inputs=(input, input_3D,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops)
    print(params)
