# **Towards More Practical Adversarial Attacks on Graph Neural Networks**

Code for our NeurIPS 2020 paper [**Towards More Practical Adversarial Attacks on Graph Neural Networks**](https://arxiv.org/abs/2006.05057) (The previous title is **Black-Box Adversarial Attacks on Graph Neural Networks with Limited Node Access**).  The code is implemented on PyTorch with python3.

### Requirements
- dgl 0.4.2
- torch 1.4.0    
- networkx 2.3  
- numpy 1.16.4 

### Running Procedure

Example command to run the code: `python main.py --dataset cora --model JKNetMaxpool --threshold 0.1 --steps 4`. Experiment settings can be specified through the python arguments.
