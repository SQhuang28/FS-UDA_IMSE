from torch import nn
from torch.nn import functional as F
import torch


class Discriminator(nn.Module):
    def __init__(self, in_features=640):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(  # 3*84*84


            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),  # 64*21*21

            nn.Linear(32, 2),
            # nn.BatchNorm1d(2),
            # nn.LeakyReLU(0.2),  # 64*21*21
        )


    def forward(self, input):
        # extract features of input1--query image
        qm, C, h, w = input.size()
        input = torch.nn.functional.adaptive_avg_pool2d(input, 1)
        q = torch.reshape(input.permute(0, 2, 3, 1), [qm, C])
        q = self.features(q)
        return q


class DiscriminatorLD(nn.Module):
    def __init__(self, in_dim):
        super(DiscriminatorLD, self).__init__()

        self.features = nn.Sequential(  # 3*84*84


            nn.Linear(in_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),  # 64*21*21


            nn.Linear(32, 2),
            nn.BatchNorm1d(2),
            nn.LeakyReLU(0.2),  # 64*21*21
        )

    def forward(self, input):
        # extract features of input1--query image
        # input = torch.nn.functional.avg_pool2d(input, 10, 10)
        qm, C, h, w = input.size()
        q = torch.reshape(input.permute(0, 2, 3, 1), [qm*h*w, C])
        q = self.features(q)
        return q