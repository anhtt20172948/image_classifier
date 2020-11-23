import torch
import torchvision
from config import *
import torch.nn as nn
from net.alexNET import AlexNet
from utils import transform
import time

if __name__ == '__main__':
    # Testing Accuracy
    correct = 0
    total = 0
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS)
    # Instantiating CUDA device
    device = torch.device("cuda:{}".format(CUDA_DEVICE) if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    model.classifier[4] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, 10)
    model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device(device)))
    print("START TESTING..........")
    start_time = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Total time test: {}".format(time.time() - start_time))
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
