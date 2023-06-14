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
```bash
$ git clone https://github.com/bath-reinforcement-learning-lab/SVERL_icml_2023.git
```

2. Install this repository and the dependencies using pip:
```bash
$ conda create --name sverl python=3.10.9  
$ conda activate sverl  
$ cd sverl_icml_2023  
$ pip install -r requirements.txt
```      
# Overview

This github enables the replication of the experiments for every domain in our paper: Shapley values applied to value functions, Shapley values applied to policies and SVERL. The Shapley values for the specified states are saved in dictionaries as pickle files. 

The dictionaries for SVERL and Shapley values applied to value functions have the following structure:

{state 1: \[Shapley value for feature 1, Shapley value for feature 2, ...\], state 2: ...}

The dictionaries for Shapley values applied to policies have the following structure:

{state 1: \[\[Shapley value for feature 1 and action 1, Shapley value for feature 1 and action 2, ...\], \[Shapley value for feature 2 and action 1, ...\]\], state 2: ...}

## Running experiment

To run an experiment, navigate to the folder named after the environment and call:

```bash
$ cd gwa/
$ python3 run.py
```

The experiments use multiprocessing. To not use multiprocessing, set the parameter "multi_process=false" when calling "local_sverl_C_values", "global_sverl_C_values", "shapley_on_policy" and "shapley_on_value" in run.py. You can also toggle the number of processes used when calling the same function with "num_p=5".

The "run.py" files are configured to calculate as many of the four types of Shapley values: "local_sverl_C_values", "global_sverl_C_values", "shapley_on_policy" and "shapley_on_value", as is computationally feasible. If you required only Shapley values used in the paper, comment out the appropriate lines.

If you need help to use SVERL, please open an issue or contact djeb20@bath.ac.uk.

# Citation

If you use find this code useful for your research, please consider citing out work:

```
@inproceedings{beechey2023explaining,
      title={Explaining Reinforcement Learning with Shapley Values}, 
      author={Daniel Beechey and Thomas M. S. Smith and Özgür Şimşek},
      year={2023},
      booktitle={ICML}
}
```
