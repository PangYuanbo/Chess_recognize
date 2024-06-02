from torchvision import models
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Dataset import CustomImageNet
from Res50 import ResNet50, Bottleneck
import torch.optim as optim
#
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#
data_path = 'C:\\Users\\14653\\Desktop\\imagenet-object-localization-challenge\\ILSVRC\\Data\\CLS-LOC\\train'
imagenet_data = CustomImageNet(data_path, transform=transform)
data_loader = DataLoader(imagenet_data, batch_size=128, shuffle=True)
print(len(data_loader))
model = ResNet50(Bottleneck, [3, 4, 6, 3], num_class=1000)
model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):
    for i, data in enumerate(data_loader):
        inputs, labels = data
        if isinstance(labels, tuple):
            labels = labels[0]  # 如果是元组，取第一个元素
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print('epoch: %d, batch: %d, loss: %.5f' % (epoch, i, loss.item()))