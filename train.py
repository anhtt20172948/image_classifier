import torch
import torchvision
import torch.nn as nn
from utils import transform

from net.alexNET import alexnet
import torch.optim as optim
from config import *
import time

def train():
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=NUM_WORKERS)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=NUM_WORKERS)

    print('==> Building model..')

    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    AlexNet_model = alexnet(pretrained=False)
    # Model description

    # Updating the second classifier
    AlexNet_model.classifier[4] = nn.Linear(4096, 1024)
    # Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output
    # nodes if we are going to get 10 class labels through our model.
    AlexNet_model.classifier[6] = nn.Linear(1024, 10)
    print(AlexNet_model)
    # Instantiating CUDA device
    device = torch.device("cuda:{}".format(CUDA_DEVICE) if torch.cuda.is_available() else "cpu")

    # Move the input and AlexNet_model to GPU for speed if available
    AlexNet_model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer default(SGD)
    if OPTIMIZER == 'Adam':
        optimizer = optim.Adam(AlexNet_model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(AlexNet_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    print("START TRAINING..........")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = AlexNet_model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training of AlexNet')
    print("Total time training: {}".format(time.time() - start_time))
    torch.save(AlexNet_model.state_dict(), SAVE_CHECKPOINT_PATH)


if __name__ == '__main__':
    train()
