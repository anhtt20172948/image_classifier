# HYPERPARAMETER for training
NUM_EPOCHES = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
SAVE_CHECKPOINT_PATH = './checkpoint/cifar_net.pth'
OPTIMIZER = 'SGD'  # choose ['SGD', 'Adam']

# For testing
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
PATH = './checkpoint/cifar_net.pth'
IMG_PATH = './images/person.jpg'
CHECKPOINT_PATH = './checkpoint/cifar_net.pth'
