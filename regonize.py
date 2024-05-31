from torchvision import transforms,datasets
from torch.utils.data import DataLoader

input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir='data/'

image_datasets = {
    'train': datasets.ImageFolder(data_dir + 'train', data_transforms['train']),
    'val': datasets.ImageFolder(data_dir + 'val', data_transforms['val'])
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=4, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=4, shuffle=False, num_workers=4)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# check if dataset is loaded correctly
for inputs, labels in dataloaders['train']:
    print(inputs.size(), labels.size())
    break