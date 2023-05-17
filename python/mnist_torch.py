from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 256),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.model(x)


def train(model, device, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)


def main():
    device = torch.device("cuda")
    batch_size = 100

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_mnist = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_mnist, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True
    )

    model = Net().to(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total = 0
    numReps = 20

    for i in range(numReps):
        print(i)
        start.record()
        train(model, device, train_loader)
        end.record()

        torch.cuda.synchronize()
        total += start.elapsed_time(end)

    print(f"One epoch took {total / numReps * 1000} microseconds on average")


if __name__ == "__main__":
    main()
