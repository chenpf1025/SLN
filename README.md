# Noise against noise: stochastic label noise helps combat inherent label noise.
This is the official repository for the paper [Noise against noise: stochastic label noise helps combat inherent label noise](https://openreview.net/forum?id=80FMcTSZ6J0). (ICLR 2021, Spotlight).
```
@inproceedings{chen2021noise,
    title={Noise against noise: stochastic label noise helps combat inherent label noise},
    author={Chen, Pengfei and Chen, Guangyong and Ye, Junjie and Zhao, Jingwei and Heng, Pheng-Ann},
    booktitle={International Conference on Learning Representations},
    year={2021}
}
```


## Overview - stochastic label noise improves generalization.
In this paper, we analyze the implicit regularization effect of stochastic label noise (SLN) and show that it can improve model performance on datasets with "inherent" label corruption. In general, SLN shall be effective when there is severe overfitting. The implementation of the standard SLN simply requires two lines of code in the training (the function train_noise in utils.py):
```
if args.sigma>0:
    target += args.sigma*torch.randn(target.size()).to(device)
```

We show that the SGD noise induced by SLN helps the model escape sharp local minima and prevents overconfident predictions, as illustrated in the figure.

<div align=center><img src="https://github.com/chenpf1025/SLN/blob/master/results/landscape.PNG" width = "100%"/></div>




## Experiments

### Requirements
* Python 3.6+
* PyTorch 1.2+
* torchvision 0.4+
* pillow 5.0+
* numpy 1.17+

### CIFAR-10
**SLN and SLN-MO**
```
python noise_cifar_train.py --sigma 1.0 --noise_mode sym --correction -1
python noise_cifar_train.py --sigma 0.5 --noise_mode asym --correction -1
python noise_cifar_train.py --sigma 0.5 --noise_mode dependent --correction -1
python noise_cifar_train.py --sigma 0.5 --noise_mode openset --correction -1
```

**SLN-MO-LC**
```
python noise_cifar_train.py --sigma 1.0 --noise_mode sym --correction 250
python noise_cifar_train.py --sigma 0.5 --noise_mode asym --correction 250
python noise_cifar_train.py --sigma 0.5 --noise_mode dependent --correction 250
python noise_cifar_train.py --sigma 0.5 --noise_mode openset --correction 250
```

### CIFAR-100
**SLN and SLN-MO**
```
python noise_cifar_train.py --sigma 0.2 --noise_mode sym --correction -1 --dataset cifar100 --num_class 100 --datapath ./data/CIFAR100
python noise_cifar_train.py --sigma 0.2 --noise_mode asym --correction -1 --dataset cifar100 --num_class 100 --datapath ./data/CIFAR100
python noise_cifar_train.py --sigma 0.1 --noise_mode dependent --correction -1 --dataset cifar100 --num_class 100 --datapath ./data/CIFAR100
```

**SLN-MO-LC**
```
python noise_cifar_train.py --sigma 0.2 --noise_mode sym --correction 250 --dataset cifar100 --num_class 100 --datapath ./data/CIFAR100
python noise_cifar_train.py --sigma 0.2 --noise_mode asym --correction 250 --dataset cifar100 --num_class 100 --datapath ./data/CIFAR100
python noise_cifar_train.py --sigma 0.1 --noise_mode dependent --correction 250 --dataset cifar100 --num_class 100 --datapath ./data/CIFAR100
```

### Clothing1M
**SLN and SLN-MO**
```
python noise_clothing1m_train.py --sigma 0.2 --correction -1
```

**SLN-MO-LC**
```
python noise_clothing1m_train.py --sigma 0.2 --correction 1
```
