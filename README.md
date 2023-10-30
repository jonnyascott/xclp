# XCLP

This repository contains the code necessary to reproduce the experiments from the paper:
## Cross-client Label Propagation for Transductive and Semi-Supervised Federated Learning


##  Data pre-processing:

### CIFAR-10
Run the following command:

```
>> cd data-local/bin
>> ./prepare_cifar10.sh
```

### CIFAR-100
Run the following command:
```
>> cd data-local/bin
>> ./prepare_cifar100.sh
```

### Mini-Imagenet
We took the Mini-Imagenet dataset hosted in [this repository](https://github.com/gidariss/FewShotWithoutForgetting) and pre-processed it.

Download [train.tar.gz](http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/train.tar.gz) and [test.tar.gz](http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/test.tar.gz), and extract them in the following directory:
```
>> ./data-local/images/miniimagenet/
```

Data preprocessing commands taken from: https://github.com/ahmetius/LP-DeepSSL.

##  Running the experiments:

FedProp consists of two stages. 

Stage 1 consists of training only on clients labeled data:
```
>> python train_stage1.py --dataset $DATASET --num_labels $NUMLABELS --iid $IID -g $GPUID
```
where ```$DATASET``` is the name of the dataset (cifar10, cifar100, or miniimagenet), ```$NUMLABELS``` is the number of labeled points (1000 or 5000 for cifar10, 5000 or 10000 for cifar100 and miniimagenet), ```$IID``` determines if client data is (iid or noniid), and ```$GPUID``` is the GPU to be used (0 by default).

After the training for Stage 1 is completed, run the following command for Stage 2, which resumes the training from the model trained in Stage 1, but this time with pseudo-labels for unlabeled data:

```
>> python train_stage2.py --dataset $DATASET --num_labels $NUMLABELS --iid $IID -g $GPUID
```
