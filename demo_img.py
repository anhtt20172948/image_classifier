from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from net.alexNET import AlexNet
from utils import transform
from config import *

if __name__ == '__main__':

    input_image = Image.open(IMG_PATH)
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    device = torch.device("cuda:{}".format(CUDA_DEVICE) if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    model.classifier[4] = nn.Linear(4096, 1024)
    model.classifier[6] = nn.Linear(1024, 10)
    model.load_state_dict(torch.load(CHECKPOINT_PATH,map_location=torch.device(device)))

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    output = torch.nn.functional.softmax(output[0], dim=0)
    res = torch.max(output).cpu().detach().numpy()
    print("images is {} format confidence {}".format(classes[torch.argmax(output)], res))
    d = ImageDraw.Draw(input_image)
    font = ImageFont.truetype('./font/abel-regular.ttf', 90)
    d.text((10, 10), '{}: {} %'.format(classes[torch.argmax(output)], round(float(res), 4) * 100),
           fill=(255, 255, 0), font=font)
    input_image.show()
