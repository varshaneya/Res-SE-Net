This consists of Pytorch implementation of ReSENet along with baseline Resnet and SE-Resnet.

This has been trained and tested on pytorch version 0.4.1. So please use the same version.
Do not use version pytorch version 1.0

There is script file by name "script.sh" provided for training.
The script automatically downloads necessary datasets, please make sure to have internet connection while running this.

The script provides automatically creates directories for storing the trained model, so you do not have to
create any.

You can modify the script to change the architecture, dataset and depth before training.

There are 3 architectures available:
1. Baseline Resnet (https://arxiv.org/pdf/1512.03385.pdf)
2. Res-SE-Net (Our proposed model)
3. SE-Resnet (https://arxiv.org/pdf/1709.01507.pdf)

Training can be done on 2 datasets:
1. CIFAR-10
2. CIFAR-100

Run this to give executable permission to the script
chmod 777 script.sh

You can then run the script by typing
./script.sh

It is advisable to retain the hyperparameters.
Please retain random seed in cifar.py for reproducibility.

### Citation

If you use Res-SE-Net model or the code please cite my work as:

     @article{res-se-net,
      title={Res-SE-Net: Boosting Performance of Resnets by Enhancing Bridge-connections},
      author={Varshaneya V, Balasubramanian S, Darshan Gera},
      journal={arXiv preprint arXiv:1902.06066},
      year={2019}
    }
