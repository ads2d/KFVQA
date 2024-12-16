import torch
import torch.nn as nn

class SelfAttention1D(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super(SelfAttention1D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        # 初始化偏置B
        self.B = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        queries = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores += self.B  # 应用偏置B

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)

        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, seq_len, -1)

        return attended_values

# 使用示例
# input_dim = 2304
# num_heads = 2
# input_data = torch.randn(16, 16, 2304)
#
# attention_module = SelfAttention1D(input_dim, num_heads)
# output = attention_module(input_data)
# print(output.shape)  # 输出形状：torch.Size([16, 16, 2304])
