import numpy as np
import torch
from torch import nn

from utils import fs


class CalModel(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        source_pos_init = torch.zeros(timesteps, 3)
        source_pos_init[:, 2] = 1000

        self.source_pos = nn.Parameter(source_pos_init)

        locations = np.zeros((192, 3))

        self.bias = nn.Parameter(torch.zeros(192))

        for idx_r, r in enumerate([34.5, 74.5, 124.5, 184, 258.315, 349.5, 461.5, 599.5]):
            for idx_a, a in enumerate(range(0, 360, 15)):
                a_rad = np.pi * a / 180
                locations[idx_a * 8 + idx_r, :2] = (np.sin(a_rad) * r, np.cos(a_rad) * r)

        self.starting_mic_pos = nn.Parameter(torch.Tensor(locations), requires_grad=False)
        self.mic_pos = nn.Parameter(torch.Tensor(locations))

        self.speed_of_sound = nn.Parameter(torch.Tensor([343e3]))  # mm / s

    def forward(self, offsets, corrs):
        # first compute the distance between the source and each mic
        dist = torch.cdist(self.source_pos, self.mic_pos)

        # then compute the distance difference between each microphone
        dist_diff = dist.unsqueeze(2) - dist.unsqueeze(1)

        # then compute the time difference
        time_diff = dist_diff / self.speed_of_sound * fs

        time_err = time_diff - offsets # + self.bias[None, None, :] - self.bias[None, :, None]

        time_err.masked_fill_(corrs == 0, 0)

        # remove outliers by removing the largest 10% errors in each row
        mask = time_err.abs() > torch.quantile(time_err.abs(), 0.9, keepdim=True)

        time_err.masked_fill_(mask, 0)
        time_err.masked_fill_(mask.transpose(1, 2), 0)

        time_err = time_err.abs().mean()

        # deviation of mic_pos from starting_mic_pos
        mic_pos_err = (self.mic_pos - self.starting_mic_pos).abs().mean()

        jerk = torch.diff(self.source_pos, dim=0, n=3).norm(dim=1).mean()

        return time_err + mic_pos_err * 0.5 + jerk * 0.01, time_err, mic_pos_err, jerk
