# **Towards More Practical Adversarial Attacks on Graph Neural Networks**

This repo provides the official implementations for the experiments described in the following paper:

[**Towards More Practical Adversarial Attacks on Graph Neural Networks**](https://arxiv.org/abs/2006.05057)

Jiaqi Ma\*, Shuangrui Ding\*, and Qiaozhu Mei. NeurIPS 2020.

(\*: equal constribution)

A previous version of this paper is appeared with the title *Black-Box Adversarial Attacks on Graph Neural Networks with Limited Node Access*.

## Requirements
- dgl 0.4.2
- torch 1.4.0    
- networkx 2.3  
- numpy 1.16.4 

## Run the code

Example command to run the code: `python main.py --dataset cora --model JKNetMaxpool --threshold 0.1 --steps 4`. 

## Cite
```
@inproceedings{ma2020practical,
  title={Towards More Practical Adversarial Attacks on Graph Neural Networks},
  author={Ma, Jiaqi and Ding, Shuangtui and Mei, Qiaozhu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
