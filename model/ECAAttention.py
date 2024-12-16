import torch
from torch import nn
from torch.nn import init

class ECAAttention1D(nn.Module):

    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)  # bs, c, 1
        y = y.permute(0, 2, 1)  # bs, 1, c
        y = self.conv(y)  # bs, 1, c
        y = self.sigmoid(y)  # bs, 1, c
        y = y.permute(0, 2, 1)  # bs, c, 1
        return x * y

if __name__ == '__main__':
    block = ECAAttention1D(in_channels=64, kernel_size=3)
    input = torch.rand(1, 64, 64)
    output = block(input)
    print(input.size(), output.size())

