import torch
import torch.nn as nn
from torch.optim import SGD


X = torch.tensor([[0.26, 0.33]]).reshape(1, 2)
Y = torch.tensor([1.0, 0.0]).reshape(1, 2)

W1 = torch.tensor([[0.1, 0.5], [0.2, 0.4], [0.4, 0.1]])
b1 = torch.tensor([0.3, 0.3, 0.3])
W2 = torch.tensor([[0.1, 0.5, 0.3], [0.2, 0.1, 0.4]])
b2 = torch.tensor([0.7, 0.7])

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 2),
)

model[0].weight = torch.nn.Parameter(W1)
model[0].bias   = torch.nn.Parameter(b1)
model[2].weight = torch.nn.Parameter(W2)
model[2].bias   = torch.nn.Parameter(b2)

optimizer = SGD(model.parameters(), lr=0.5)
criterion = nn.CrossEntropyLoss()


for i in range(1):
    model.zero_grad()
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()

print('Prediction:', pred)
print('Loss:', loss)
print('=============')
print('W1', model[0].weight)
print('W2', model[2].weight)
print('b1', model[0].bias)
print('b2', model[2].bias)
