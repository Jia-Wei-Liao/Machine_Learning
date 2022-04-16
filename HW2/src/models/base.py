import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def save(self, path):
        torch.save(self, path)

        return

    def load(self, path):
        self = torch.load(path)

        return
