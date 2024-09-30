# Introduction

RL4CO is an extensive Reinforcement Learning (RL) for Combinatorial Optimization (CO) benchmark. Our goal is to provide a unified framework for RL-based CO algorithms, and to facilitate reproducible research in this field, decoupling the science from the engineering.




## Motivation

### Why NCO?
Neural Combinatorial Optimization (NCO) is a subfield of AI that aims to solve combinatorial optimization problems using neural networks. NCO has been successfully applied to a wide range of problems, such as the routing problems in logistics, the scheduling problems in manufacturing, and electronic design automation. The key idea behind NCO is to learn a policy that maps the input data to the optimal solution, without the need for hand-crafted heuristics or domain-specific knowledge.


### Why RL?
Reinforcement Learning (RL) is a machine learning paradigm that enables agents to learn how to make decisions by interacting with an environment. RL has been successfully applied to a wide range of problems, such as playing games, controlling robots, and optimizing complex systems. The key idea behind RL is to learn a policy that maps the state of the environment to the optimal action, by maximizing a reward signal. Importantly, optimal solutions are not required for training, as RL agents learn from the feedback they receive from the environment.



## Contents

We explore in other pages the following components:

- [Environments](environments.md): Markov Decision Process (MDP) for CO problems and base classes for environments. These are based on [TorchRL](https://pytorch.org/rl/stable/index.html).

- [Policies](policies.md): the neural networks that are used to solve CO problems and their base classes. These are based on [PyTorch](https://pytorch.org/).

- [RL Algorithms](rl.md): (broadly: "models"), which are the processes used to train the policies and their base classes. These are based on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).


<div align="center">
  <img src="https://github.com/ai4co/rl4co/assets/48984123/0e409784-05a9-4799-b7aa-6c0f76ecf27f" alt="RL4CO-Overview" style="max-width: 90%;">
</div>


## Paper Reference

Our paper is available [here](https://arxiv.org/abs/2306.17100) for further details.

If you find RL4CO valuable for your research or applied projects, don't forget to cite us! 🚀

```bibtex
@article{berto2024rl4co,
    title={{RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark}},
    author={Federico Berto and Chuanbo Hua and Junyoung Park and Laurin Luttmann and Yining Ma and Fanchen Bu and Jiarui Wang and Haoran Ye and Minsu Kim and Sanghyeok Choi and Nayeli Gast Zepeda and Andr\'e Hottung and Jianan Zhou and Jieyi Bi and Yu Hu and Fei Liu and Hyeonah Kim and Jiwoo Son and Haeyeon Kim and Davide Angioni and Wouter Kool and Zhiguang Cao and Jie Zhang and Kijung Shin and Cathy Wu and Sungsoo Ahn and Guojie Song and Changhyun Kwon and Lin Xie and Jinkyoo Park},
    year={2024},
    journal={arXiv preprint arXiv:2306.17100},
    note={\url{https://github.com/ai4co/rl4co}}
}
```
