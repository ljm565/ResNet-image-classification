import torch
import os
from PIL import Image
from tqdm import tqdm
from model_ResNet import ResNet, ResidualBlock, CNNBlock


def save_checkpoint(file, model, optimizer, scheduler):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')


def make_img_data(path, trans):
    files = os.listdir(path)
    data = [trans(Image.open(path+file)) for file in tqdm(files) if not file.startswith('.')]
    return data


def model_select(config, color_channel, device):
    if config.model_mode.lower() == 'cnn':
        print('CNN{} will be trained..'.format(int(config.num_layer*2*3+2)))
        block = CNNBlock
        model = ResNet(config, color_channel, config.num_layer, block)
    elif config.model_mode.lower() == 'resnet':
        print('ResNet{} will be trained..'.format(int(config.num_layer*2*3+2)))
        block = ResidualBlock
        model = ResNet(config, color_channel, config.num_layer, block)
    else:
        print("model mode have to be cnn or resnet")
        raise AssertionError
    return model.to(device)
