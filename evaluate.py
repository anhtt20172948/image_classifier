import torch
import torchvision
from utils import transform
import torch.nn as nn
from net.alexNET import AlexNet
from config import *


if __name__ == '__main__':
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=0)
    # Instantiating CUDA device
    device = torch.device("cuda:{}".format(CUDA_DEVICE) if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    model.to(device)
    model.classifier[4] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, 10)
    model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))