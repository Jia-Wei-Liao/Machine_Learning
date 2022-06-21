import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EfficientNet_b4
from src.models.build import build_model
from src.models.swin_utils import load_pretrained


class TeacherStudentModel(nn.Module):
    def __init__(self, num_classes=100):
        super(TeacherStudentModel, self).__init__()
        self.student_model = EfficientNet_b4(num_classes)
        self.teacher_model = torch.load("checkpoint/04-18-02-12-47/weight/ep=0043-acc=0.9360.pth")

    def save(self, path):
        torch.save(self, path)

        return

    def forward(self, inputs, *args, **kwargs):
        with torch.no_grad():
            teacher_out = self.teacher_model(inputs)

        student_out = self.student_model(inputs)

        return teacher_out, student_out


if __name__ == '__main__':
    print(model)
