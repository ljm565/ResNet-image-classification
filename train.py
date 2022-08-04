import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import copy
import pickle
import math
import os 
import numpy as np

from config import Config
from utils_func import model_select, save_checkpoint, make_img_data
from utils_data import DLoader



class Trainer:
    def __init__(self, config:Config, device:torch.device, mode:str, continuous:int):
        self.config = config
        self.device = device
        self.mode = mode
        self.continuous = continuous
        self.dataloaders = {}

        # if continuous, load previous training info
        if self.continuous:
            with open(self.config.loss_data_path, 'rb') as f:
                self.loss_data = pickle.load(f)

        # path, data params
        self.base_path = self.config.base_path
        self.model_path = self.config.model_path
        self.color_channel = self.config.color_channel
        assert self.color_channel in [1, 3]
        self.convert2grayscale = True if self.color_channel==3 and self.config.convert2grayscale else False
        self.color_channel = 1 if self.convert2grayscale else self.color_channel

        # train params
        self.batch_size = self.config.batch_size
        self.steps = self.config.steps
        self.lr = self.config.lr


        # split trainset to trainset and valset and make dataloaders
        if self.config.CIFAR10_train:
            # for reproducibility
            torch.manual_seed(999)

            # set to CIFAR10 size
            self.config.width, self.config.height, self.config.label, self.color_channel = 32, 32, 10, 3

            # set transforms
            self.trans_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
            self.trans_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            self.CIFAR10_valset_proportion = self.config.CIFAR10_valset_proportion
            self.trainset = dsets.CIFAR10(root=self.base_path, train=True, download=True, transform=self.trans_train)
            self.trainset, self.valset = random_split(self.trainset, [len(self.trainset)-int(len(self.trainset)*self.CIFAR10_valset_proportion), int(len(self.trainset)*self.CIFAR10_valset_proportion)])
            self.testset = dsets.CIFAR10(root=self.base_path, train=False, download=True, transform=self.trans_test)
        else:
            os.makedirs(self.base_path+'data', exist_ok=True)

            if os.path.isdir(self.base_path+'data/'+self.config.data_name):
                with open(self.base_path+'data/'+self.config.data_name+'/train.pkl', 'rb') as f:
                    self.trainset = pickle.load(f)
                with open(self.base_path+'data/'+self.config.data_name+'/val.pkl', 'rb') as f:
                    self.valset = pickle.load(f)
                with open(self.base_path+'data/'+self.config.data_name+'/test.pkl', 'rb') as f:
                    self.testset = pickle.load(f)
            else:
                os.makedirs(self.base_path+'data/'+self.config.data_name, exist_ok=True)
                self.trans = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                transforms.Resize((self.config.height, self.config.width)),
                                                transforms.ToTensor()]) if self.convert2grayscale else \
                            transforms.Compose([transforms.Resize((self.config.height, self.config.width)),
                                                transforms.ToTensor()]) 
                self.custom_data_proportion = self.config.custom_data_proportion
                assert math.isclose(sum(self.custom_data_proportion), 1)
                assert len(self.custom_data_proportion) <= 3
                
                if len(self.custom_data_proportion) == 3:
                    data = make_img_data(self.config.train_data_path, self.trans)
                    self.train_len, self.val_len = int(len(data)*self.custom_data_proportion[0]), int(len(data)*self.custom_data_proportion[1])
                    self.test_len = len(data) - self.train_len - self.val_len
                    self.trainset, self.valset, self.testset = random_split(data, [self.train_len, self.val_len, self.test_len], generator=torch.Generator().manual_seed(999))

                elif len(self.custom_data_proportion) == 2:
                    data1 = make_img_data(self.config.train_data_path, self.trans)
                    data2 = make_img_data(self.config.test_data_path, self.trans)
                    if self.config.two_folders == ['train', 'val']:
                        self.train_len = int(len(data1)*self.custom_data_proportion[0]) 
                        self.val_len = len(data1) - self.train_len
                        self.trainset, self.valset = random_split(data1, [self.train_len, self.val_len], generator=torch.Generator().manual_seed(999))
                        self.testset = data2
                    elif self.config.two_folders == ['val', 'test']:
                        self.trainset = data1
                        self.val_len = int(len(data2)*self.custom_data_proportion[0]) 
                        self.test_len = len(data2) - self.val_len
                        self.valset, self.testset = random_split(data2, [self.val_len, self.test_len], generator=torch.Generator().manual_seed(999))
                    else:
                        print("two folders must be ['train', 'val] or ['val', 'test']")
                        raise AssertionError

                elif len(self.custom_data_proportion) == 1:
                    self.trainset = make_img_data(self.config.train_data_path, self.trans)
                    self.valset = make_img_data(self.config.val_data_path, self.trans)
                    self.testset = make_img_data(self.config.test_data_path, self.trans)
                
                with open(self.base_path+'data/'+self.config.data_name+'/train.pkl', 'wb') as f:
                    pickle.dump(self.trainset, f)
                with open(self.base_path+'data/'+self.config.data_name+'/val.pkl', 'wb') as f:
                    pickle.dump(self.valset, f)
                with open(self.base_path+'data/'+self.config.data_name+'/test.pkl', 'wb') as f:
                    pickle.dump(self.testset, f)

            self.trainset, self.valset, self.testset = DLoader(self.trainset), DLoader(self.valset), DLoader(self.testset)
        
        self.dataloaders['train'] = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.dataloaders['val'] = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        if self.mode == 'test':
            self.dataloaders['test'] = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.model = model_select(self.config, self.color_channel, self.device)
        self.criterion = nn.CrossEntropyLoss()
        if self.mode == 'train':
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            decay_steps = [32000, 48000]
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=decay_steps, gamma=0.1)
            if self.continuous:
                self.check_point = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(self.check_point['model'])
                self.optimizer.load_state_dict(self.check_point['optimizer'])
                self.scheduler.load_state_dict(self.check_point['scheduler'])
                del self.check_point
                torch.cuda.empty_cache()
        elif self.mode == 'test':
            self.check_point = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(self.check_point['model'])
            self.model.eval()
            del self.check_point
            torch.cuda.empty_cache()

        
    def train(self):
        best_val_loss = float('inf') if not self.continuous else self.loss_data['best_val_loss']
        train_loss_history = [] if not self.continuous else self.loss_data['train_loss_history']
        val_loss_history = [] if not self.continuous else self.loss_data['val_loss_history']
        train_acc_history = [] if not self.continuous else self.loss_data['train_acc_history']
        val_acc_history = [] if not self.continuous else self.loss_data['val_acc_history']
        best_epoch_info = 0 if not self.continuous else self.loss_data['best_epoch']

        step, epoch = 0, 0
        while step < self.steps:
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss, total_acc = 0, 0
                for x, y in self.dataloaders[phase]:
                    batch = x.size(0)
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=='train'):
                        output = self.model(x)
                        loss = self.criterion(output, y)
                        acc = (torch.argmax(output, dim=1) == y).float().sum()/batch

                        if phase == 'train':
                            step += 1
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                    total_loss += loss.item()*batch
                    total_acc += acc.item()*batch
                    if step % 1000 == 0 or step == 1:
                        print('Epoch {}: {}/{} step loss: {}, step acc: {}'.format(epoch+1, step, 64000, loss.item(), acc.item()))
                epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
                epoch_acc = total_acc/len(self.dataloaders[phase].dataset)
                print('Epoch {}: {}/{} step loss: {}, step acc: {}'.format(epoch+1, step, 64000, epoch_loss, epoch_acc))

                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                if phase == 'val':
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        best_epoch = best_epoch_info + epoch + 1
                        save_checkpoint(self.model_path, self.model, self.optimizer, self.scheduler)
            epoch += 1
            
        print('best val loss: {:4f}, best epoch: {:d}'.format(best_val_loss, best_epoch))
        self.model.load_state_dict(best_model_wts)
        self.loss_data = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history, 'train_acc_history': train_acc_history, 'val_acc_history': val_acc_history}
        return self.model, self.loss_data
    

    def test(self, result_num):
        if result_num > len(self.dataloaders['test'].dataset):
            print('The number of results that you want to see are larger than total test set')
            raise AssertionError
        
        # predict CIFAR-10 test set
        phase = 'test'
        all_data, gt, ids = [], [], set()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            self.model.eval()
            for x, y in self.dataloaders[phase]:
                batch = x.size(0)
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x)
                loss = self.criterion(output, y)
                acc = (torch.argmax(output, dim=1) == y).float().sum()/batch

                total_loss += loss.item()*batch
                total_acc += acc.item()*batch

                all_data.append(x.detach().cpu())
                gt.append(y.detach().cpu())

            all_data = torch.cat(all_data, dim=0)
            gt = torch.cat(gt, dim=0)
            epoch_loss = total_loss/len(self.dataloaders[phase].dataset)
            epoch_acc = total_acc/len(self.dataloaders[phase].dataset)
            print('{} loss: {:4f}, acc: {:4f}\n'.format(phase, epoch_loss, epoch_acc))

        
        
        while 1:
            if len(ids) == result_num:
                break
            id = np.random.randint(1, high=all_data.size(0))
            ids.add(id)
        ids = list(ids)
        test_samples = torch.cat([all_data[id].unsqueeze(0) for id in ids], dim=0).to(self.device)
        test_samples_gt = torch.cat([gt[id].unsqueeze(0) for id in ids], dim=0).to(self.device)
        output = self.model(test_samples)
        output = torch.argmax(output, dim=1)
        print('ground truth: {}'.format(test_samples_gt))
        print('prediction: {}'.format(output))