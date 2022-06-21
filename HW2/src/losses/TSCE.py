import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TSCE_Loss(nn.Module):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def forward(self, student_out, teacher_out):
        student_out = torch.softmax(student_out, dim=-1)
        teacher_out = torch.softmax(teacher_out, dim=-1)
        loss = -(teacher_out * torch.log(student_out + self.epsilon)).sum(dim=1).mean()

        return loss


def TSCE(student_out, teacher_out):
    student_out = torch.softmax(student_out, dim=-1)
    teacher_out = torch.softmax(teacher_out, dim=-1)
    loss = -(teacher_out * torch.log(student_out + 1e-8)).sum(dim=1).mean()

    return loss


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    mcloss = MCCE_Loss()
    targets = torch.from_numpy(np.arange(2)).long()
    feat = torch.randn((2, 1792, 13, 13))
    loss = mcloss(feat, F.one_hot(targets).float(), targets)
    print(loss)
