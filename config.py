#  for training
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
SAVE_CHECKPOINT_PATH = './checkpoint'
OPTIMIZER = 'SGD'  # choose ['SGD', 'Adam']
NUM_WORKERS = 0
BATCH_SIZE = 4
CUDA_DEVICE = '0'

# For testing
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
IMG_PATH = './images/bird.jpg'
CHECKPOINT_PATH = './checkpoint/cifar_net.pth'
