# Explaining Reinforcement Learning with Shapley Values

Official implementation of Shapley Values for Explaining Reinforcement Learning (SVERL), ICML 2023 | [paper](https://arxiv.org/abs/2306.05810)

Daniel Beechey, Thomas M. S. Smith, Özgür Şimşek

## Abstract

For reinforcement learning systems to be widely adopted, their users must understand and trust them. We present a theoretical analysis of explaining reinforcement learning using Shapley values, following a principled approach from game theory for identifying the contribution of individual players to the outcome of a cooperative game. We call this general framework Shapley Values for Explaining Reinforcement Learning (SVERL). Our analysis exposes the limitations of earlier uses of Shapley values in reinforcement learning. We then develop an approach that uses Shapley values to explain agent performance. In a variety of domains, SVERL produces meaningful explanations that match and supplement human intuition.

# Installation

## Requirements overview

- cloudpickle==2.2.1
- colorama==0.4.6
- gym==0.26.2
- gym-notices==0.0.8
- numpy==1.24.3
- tqdm==4.65.0

## Procedure

1. Clone the repo:

      `$ git clone https://github.com/bath-reinforcement-learning-lab/SVERL_icml_2023.git`
      
2. Install this repository and the dependencies using pip:

      `conda create --name sverl python=3.10.9  
      conda activate sverl  
      cd sverl_icml_2023  
      pip install -r requirements.txt`
      
# Overview

# Conclusion

If you need help to use SVERL, please open an issue or contact djeb20@bath.ac.uk.

# Citation

If you use find this code useful for your research, please consider citing out work:

@inproceedings{beechey2023explaining,
      title={Explaining Reinforcement Learning with Shapley Values}, 
      author={Daniel Beechey and Thomas M. S. Smith and Özgür Şimşek},
      year={2023},
      booktitle={ICML}
}
