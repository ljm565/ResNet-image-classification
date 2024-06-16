# Training ResNet Image Classification
여기서는 ResNet 이미지 분류 모델을 학습하는 가이드를 제공합니다.

### 1. Configuration Preparation
ResNet 이미지 분류 모델을 학습하기 위해서는 Configuration을 작성하여야 합니다.
Configuration에 대한 option들의 자세한 설명 및 예시는 다음과 같습니다.

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


# prediction result visualzation config
result_num: 10              # number of prediction results to be shown

# logging config
common: ['train_loss', 'train_acc', 'validation_loss', 'validation_acc', 'lr']
```


### 2. Training
#### 2.1 Arguments
`src/run/train.py`를 실행시키기 위한 몇 가지 argument가 있습니다.
* [`-c`, `--config`]: 학습 수행을 위한 config file 경로.
* [`-m`, `--mode`]: [`train`, `resume`] 중 하나를 선택.
* [`-r`, `--resume_model_dir`]: mode가 `resume`일 때 모델 경로. `{$project}/{$name}`까지의 경로만 입력하면, 자동으로 `{$project}/{$name}/weights/`의 모델을 선택하여 resume을 수행.
* [`-l`, `--load_model_type`]: [`metric`, `loss`, `last`] 중 하나를 선택.
    * `metric`(default): Valdiation accuracy가 최대일 때 모델을 resume.
    * `loss`: Valdiation loss가 최소일 때 모델을 resume.
    * `last`: Last epoch에 저장된 모델을 resume.
* [`-p`, `--port`]: (default: `10001`) DDP 학습 시 NCCL port.


#### 2.2 Command
`src/run/train.py` 파일로 다음과 같은 명령어를 통해 ResNet 모델을 학습합니다.
```bash
# training from scratch
python3 src/run/train.py --config configs/config.yaml --mode train

# training from resumed model
python3 src/run/train.py --config config/config.yaml --mode resume --resume_model_dir {$project}/{$name}
```
모델 학습이 끝나면 `{$project}/{$name}/weights`에 체크포인트가 저장되며, `{$project}/{$name}/args.yaml`에 학습 config가 저장됩니다.
그리고 `config/config.yaml`에 세팅 된 값을 바탕으로 학습 lr curve 가시화 이미지가 `{$project}/{$name}/vis_outputs/lr_schedule.png`에 자동으로 저장됩니다.