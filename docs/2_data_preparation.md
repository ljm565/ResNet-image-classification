# Data Preparation
Here, we will proceed with a ResNet image classification model training tutorial using the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset by default.
Please refer to the following instructions to utilize custom datasets.


### 1. CIFAR-10
If you want to train on the CIFAR-10 dataset, simply set the `CIFAR10_train` value in the `config/config.yaml` file to `True` as follows.
```yaml
CIFAR10_train: True       
class_num: 10
CIFAR10:
    path: data/
    CIFAR10_valset_proportion: 0.2 
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
<br>

### 2. Custom Data
If you want to train your custom dataset, set the `CIFAR10_train` value in the `config/config.yaml` file to `False` as follows.
You may require to implement your custom dataloader codes in `src/utils/data_utils.py`.
```yaml
CIFAR10_train: False    
class_num: ${NUMBER_OF_CUSTOM_DATA_CLASSES}           
CIFAR10:
    path: data/
    CIFAR10_valset_proportion: 0.2 
CUSTOM:
    train_data_path: null
    validation_data_path: null
    test_data_path: null
```
