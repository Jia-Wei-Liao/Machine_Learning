from os import path
if path.exists("model.py"):
	from model import *
from torchvision import transforms, datasets
import torch
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fpth = glob.glob('*.txt')
print(fpth)
if not fpth:
	mean = (0.5071, 0.4867, 0.4408) 
	std = (0.2675, 0.2565, 0.2761)
	
else:
	with open(fpth[0]) as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if i == 0:
				mean = (float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2].replace('\n', '')))
			else:
				std = (float(line.split(' ')[0]), float(line.split(' ')[1]), float(line.split(' ')[2].replace('\n', '')))


print(mean, std)
test_transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize(mean, std)])

testset = datasets.CIFAR100(root='./data', train=False,
download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
shuffle=False, num_workers=2)

mpth = glob.glob('*.pth')
model = torch.load(mpth[0])
model.to(device)
model.eval() # add this command

# fixed testing process
correct = 0
total = 0
with torch.no_grad():
	for data in testloader:
		images, labels = data
		images = images.to(device)
		labels = labels.to(device)
		outputs = model(images)
		# the class with the highest energy is what we choose as prediction
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
	print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')