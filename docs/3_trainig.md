# Training ResNet Image Classification
Here, we provide guides for training a ResNet image classification model.

### 1. Configuration Preparation
To train a ResNet image classification model, you need to create a configuration.
Detailed descriptions and examples of the configuration options are as follows.

```yaml
# base
seed: 0
deterministic: True

# environment config
device: cpu     # examples: [0], [0,1], [1,2,3], cpu, mps... 

# project config
project: outputs/resnet_wo_zeropad
name: CIFAR10

# model config
model_type: resnet        # [resnet, cnn] If resnet, ResNet will be loaded, else, vanilla CNN will be loaded.
num_layer: 3              # Number of each residual block's layer. A model consisting of a total of (num_layer * 2 * 3 + 2) layers will be created.
zero_padding: False       # For downsampling in the shortcuts, use zero padding if True, and a 1x1 convolutional layer if False.

# image setting config
height: 32
width: 32
color_channel: 3
convert2grayscale: False

# data config
workers: 0               # Don't worry to set worker. The number of workers will be set automatically according to the batch size.
CIFAR10_train: True      # if True, CIFAR10 will be loaded automatically.
class_num: 10            # Number of image label classes.
CIFAR10:
    path: data/
    CIFAR10_valset_proportion: 0.2      # CIFAR10 has only train and test data. Thus, part of the training data is used as a validation set.
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null

# train config
batch_size: 128
steps: 64000
warmup_steps: 100
lr0: 0.001
lrf: 0.001                  # last_lr = lr0 * lrf
scheduler_type: 'cosine'    # ['linear', 'cosine']
momentum: 0.9
weight_decay: 0.0
warmup_momentum: 0.8

# logging config
common: ['train_loss', 'train_acc', 'validation_loss', 'validation_acc', 'lr']
```


### 2. Training
#### 2.1 Arguments
There are several arguments for running `src/run/train.py`:
* [`-c`, `--config`]: Path to the config file for training.
* [`-m`, `--mode`]: Choose one of [`train`, `resume`].
* [`-r`, `--resume_model_dir`]: Path to the model directory when the mode is resume. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to resume.
* [`-l`, `--load_model_type`]: Choose one of [`metric`, `loss`, `last`].
    * `metric` (default): Resume the model with the best validation set's accuracy.
    * `loss`: Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-p`, `--port`]: (default: `10001`) NCCL port for DDP training.


#### 2.2 Command
`src/run/train.py` file is used to train the model with the following command:
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir ${project}/${name}
```
When training started, the learning rate curve will be saved in `${project}/${name}/vis_outputs/lr_schedule.png` automatically based on the values set in `config/config.yaml`.
When the model training is complete, the checkpoint is saved in `${project}/${name}/weights` and the training config is saved at `${project}/${name}/args.yaml`.
