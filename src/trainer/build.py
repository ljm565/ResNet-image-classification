import os

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, distributed, random_split

from models import ResNet
from models.nn import ResidualBlock, CNNBlock
from utils import RANK, LOGGER, colorstr
from utils.data_utils import DLoader, seed_worker

PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders



def get_model(config, device):
    if config.model_type.lower() == 'cnn':
        LOGGER.info(colorstr('CNN{} will be trained..'.format(int(config.num_layer*2*3+2))))
        block = CNNBlock
        model = ResNet(config, config.num_layer, block)
    elif config.model_type.lower() == 'resnet':
        LOGGER.info(colorstr('ResNet{} will be trained..'.format(int(config.num_layer*2*3+2))))
        block = ResidualBlock
        model = ResNet(config, config.num_layer, block)
    else:
        LOGGER.info(colorstr("red", "Model_mode of the config have to be cnn or resnet. Please check your config file.."))
        raise AssertionError

    return model.to(device)


def build_dataset(config, modes):
    dataset_dict = {}
    if config.CIFAR10_train:
        # set to CIFAR10 size
        config.width, config.height = 32, 32
        config.class_num = 10 
        config.color_channel = 3

        # set augmentations
        train_aug = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        test_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        # init train, validation, test sets
        cifar10_path = config.CIFAR10.path
        cifar10_valset_proportion = config.CIFAR10.CIFAR10_valset_proportion
        trainset = dsets.CIFAR10(root=cifar10_path, train=True, download=True, transform=train_aug)
        valset_l = int(len(trainset) * cifar10_valset_proportion)
        trainset_l = len(trainset) - valset_l
        trainset, valset = random_split(trainset, [trainset_l, valset_l])
        testset = dsets.CIFAR10(root=cifar10_path, train=False, download=True, transform=test_aug)
        tmp_dsets = {'train': trainset, 'validation': valset, 'test': testset}
        for mode in modes:
            dataset_dict[mode] = tmp_dsets[mode]
    else:
        for mode in modes:
            dataset_dict[mode] = DLoader(config.CUSTOM.get(f'{mode}_data_path'))
    return dataset_dict


def build_dataloader(dataset, batch, workers, shuffle=True, is_ddp=False):
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if not is_ddp else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return DataLoader(dataset=dataset,
                              batch_size=batch,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              collate_fn=getattr(dataset, 'collate_fn', None),
                              worker_init_fn=seed_worker,
                              generator=generator)


def get_data_loader(config, modes, is_ddp=False):
    datasets = build_dataset(config, modes)
    dataloaders = {m: build_dataloader(datasets[m], 
                                       config.batch_size, 
                                       config.workers, 
                                       shuffle=(m == 'train'), 
                                       is_ddp=is_ddp) for m in modes}

    return dataloaders