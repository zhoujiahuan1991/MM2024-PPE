# Progressive Prototype Evolving for Dual-Forgetting Mitigation in Non-Exemplar Online Continual Learning

Official implementation of "[Progressive Prototype Evolving for Dual-Forgetting Mitigation in Non-Exemplar Online Continual Learning](https://openreview.net/forum?id=y5R8XVVA03)"


<p align="center"><img src="./files/pipeline-ppe.png" align="center" width="750"></p>


## Requirements

### Environment
Python 3.9.18

PyTorch 1.13.1+cu117

### Datasets
CIFAR-10 and CIFAR-100 will be automatically download.
Download the MiniImageNet Datasets from [this link](https://www.kaggle.com/whitemoon/miniimagenet/download).
## Run commands


```shell
# modify the --dataset_dir with your data path
bash r-cifar10.sh # Results on CIFAR-10
bash r-cifar100.sh # Results on CIFAR-100
bash r-mini.sh # Results on MiniImageNet
```

## Acknowledgement

This project is mainly based on [OnPro](https://github.com/weilllllls/OnPro), [GPM](https://github.com/sahagobinda/GPM), and [PCR](https://github.com/FelixHuiweiLin/PCR).

## Citation

If you find this work helpful, please cite:
```
@inproceedings{li2024progressive,
  title={Progressive Prototype Evolving for Dual-Forgetting Mitigation in Non-Exemplar Online Continual Learning},
  author={Li, Qiwei and Peng, Yuxin and Zhou, Jiahuan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={2477--2486},
  year={2024}
}

```

## Contact

Welcome to our Laboratory Homepage ([OV<sup>3</sup> Lab](https://zhoujiahuan1991.github.io/)) for more information about our papers, source codes, and datasets.
