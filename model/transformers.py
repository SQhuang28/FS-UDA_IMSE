import torch, math
from torch import nn

class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttentionBlock, self).__init__()
        self.output_dim = output_dim
        self.Q = nn.Linear(input_dim, output_dim)
        self.K = nn.Linear(input_dim, output_dim)
        self.V = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if len(x.size()) != 2:
            raise ValueError("Input feature should be a 2 dim tensor, your is {} dim".format(len(x.size())))
        q, k, v = self.Q(x), self.K(x), self.V(x)
        attention_map = (q@k.t()/math.sqrt(self.output_dim)).softmax(-1)
        v = v + attention_map@v
        return v

class TaskTransoformerBlock(nn.Module):
    def __init__(self, ld_num, dsp_scale=2, input_dim=640, head_num=4):
        super(TaskTransoformerBlock, self).__init__()
        self.attention_heads = nn.ModuleList()
        self.downsampling = nn.Linear(ld_num, ld_num//dsp_scale)
        self.head_num = head_num
        for _ in range(head_num):
            self.attention_heads.append(SelfAttentionBlock(input_dim, 16))

    def forward(self, x):
        V = []
        for i in range(self.head_num):
            V.append(self.attention_heads[i](x))
        # 1, ld_num, heda_num * 16
        V = torch.cat(V, dim=-1).unsqueeze(0)
        # 1, heda_num * 16, ld_num X ld_num, ld_num//dsp_scale
        V = self.downsampling(V.permute(0, 2, 1)).squeeze().t()
        return V

class MaskEncoder(nn.Module):
    def __init__(self, ld_num=100, input_dim=640, way_num=5, shot_num=1):
        super(MaskEncoder, self).__init__()
        ori_feat_num = ld_num*way_num*shot_num
        self.mask_size = [way_num, shot_num*ld_num]
        self.encoder = nn.Sequential(
            TaskTransoformerBlock(ori_feat_num, 4, input_dim, head_num=4),
            TaskTransoformerBlock(ori_feat_num//4, 2, 64, head_num=4),
            TaskTransoformerBlock(ori_feat_num//8, 2, 64, head_num=4),
        )
        self.mask_gen = nn.Linear(64*(ori_feat_num//(4*2*2)), way_num*shot_num*ld_num)

    def forward(self, x):
        x = self.encoder(x).contiguous().view(1, -1)
        x = self.mask_gen(x).view(*self.mask_size)
        return x


