import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.datasets import ImageFolder
import argparse
import torch.optim as optim
import os
import torch
from torch.utils.data import DataLoader
from datetime import datetime

# Extract the arguments from the CLI

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--resnet_size', type=int, default=50, help='ResNet network number of layers')
parser.add_argument('--max_samples_per_class', type=int, default=1000, help='Maximum number of samples per class')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--data_dir', type=str, default="data/images/", help='Base directory for images')
parser.add_argument('--output_dir', type=str, default="output", help='Output directory for the model')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train the model')
args = parser.parse_args()

# The number of classes is determined by the number of folders in the data directory
save_model = False

TRAIN_DIR = os.path.join(args.data_dir, "train")
TEST_DIR = os.path.join(args.data_dir, "test")

num_classes = len(os.listdir(TRAIN_DIR))
if num_classes == 0:
    raise ValueError("No classes found in the data directory")


# Prepare the model and the dataset, as it depends on the ResNet size
dataset = None
test_dataset = None
model = None
if args.resnet_size == 18:
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    dataset = ImageFolder(TRAIN_DIR, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
    test_dataset = ImageFolder(TEST_DIR, transform=ResNet18_Weights.IMAGENET1K_V1.transforms())
elif args.resnet_size == 34:
    model = resnet34(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    dataset = ImageFolder(TRAIN_DIR, transform=ResNet34_Weights.IMAGENET1K_V1.transforms())
    test_dataset = ImageFolder(TEST_DIR, transform=ResNet34_Weights.IMAGENET1K_V1.transforms())
elif args.resnet_size == 50:
    model = resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    dataset = ImageFolder(TRAIN_DIR, transform=ResNet50_Weights.IMAGENET1K_V1.transforms())
    test_dataset = ImageFolder(TEST_DIR, transform=ResNet50_Weights.IMAGENET1K_V1.transforms())
else:
    raise ValueError("Invalid ResNet size")

# Freeze the model parameters, to only train the last layer
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# Count the number of samples per class and limit it to max_samples_per_class
class_counts = {}
for i in range(len(dataset)):
    _, label = dataset[i]
    if label not in class_counts:
        class_counts[label] = 0
    class_counts[label] += 1
    if class_counts[label] > args.max_samples_per_class:
        dataset.samples[i] = (dataset.samples[i][0], -1)

# Drop the samples with label 0
dataset.samples = [sample for sample in dataset.samples if sample[1] != -1]

train_set, test_set, val_set = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

dataloaders = {
    "train": DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
    "test": DataLoader(test_set, batch_size=args.batch_size, shuffle=False),
    "val": DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
}

metrics = {
    'train': {
         'loss': [], 'accuracy': []
    },
    'val': {
         'loss': [], 'accuracy': []
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.resnet_size == 18:
    model = resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
elif args.resnet_size == 34:
    model = resnet34(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
elif args.resnet_size == 50:
    model = resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
else:
    raise ValueError("Invalid ResNet size")

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

model.train()
for epoch in range(args.num_epochs):
    ep_metrics = {
        'train': {'loss': 0, 'accuracy': 0, 'count': 0},
        'val': {'loss': 0, 'accuracy': 0, 'count': 0},
    }
    if save_model:
        print(f'Epoch {epoch}')
    for phase in ['train', 'val']:
        if save_model:
            print(f'-------- {phase} --------')
        for images, labels in dataloaders[phase]:
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            output = model(images.to(device))

            loss = criterion(output, labels.to(device))

            correct_preds = labels.to(device) == torch.argmax(output, dim=1)
            accuracy = (correct_preds).sum()/len(labels)

        if phase == 'train':
            loss.backward()
            optimizer.step()

        ep_metrics[phase]['loss'] += loss.item()
        ep_metrics[phase]['accuracy'] += accuracy.item()
        ep_metrics[phase]['count'] += 1

        ep_loss = ep_metrics[phase]['loss']/ep_metrics[phase]['count']
        ep_accuracy = ep_metrics[phase]['accuracy']/ep_metrics[phase]['count']

        if save_model:
            print(f'Loss: {ep_loss}, Accuracy: {ep_accuracy}\n')

        metrics[phase]['loss'].append(ep_loss)
        metrics[phase]['accuracy'].append(ep_accuracy)

date = datetime.now().strftime("%Y-%m-%d-%H-%M")
if save_model:
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"resnet{args.resnet_size}_{date}.pth"))

# test the model on data/test
model.eval()
test_loss = 0
test_accuracy = 0

test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

for images, labels in test_dataloader:
    output = model(images.to(device))
    loss = criterion(output, labels.to(device))

    correct_preds = labels.to(device) == torch.argmax(output, dim=1)
    accuracy = (correct_preds).sum()/len(labels)

    test_loss += loss.item()
    test_accuracy += accuracy.item()

test_loss /= len(test_dataloader)
test_accuracy /= len(test_dataloader)

print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save model
model.save_pretrained(args.output_dir)