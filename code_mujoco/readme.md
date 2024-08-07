# Regret Based Defense for Adversarially Robust Reinforcement Learning



## Abstract

Deep Reinforcement Learning (DRL) policies are vulnerable to adversarial noise in observations, which can have disastrous consequences in safety-critical environments. For instance, a self-driving car receiving adversarially perturbed sensory observations about traffic signs (e.g., a stop sign physically altered to be perceived as a speed limit sign) can be fatal. Leading existing approaches for making RL algorithms robust to an observation-perturbing adversary have focused on (a) regularization approaches that make expected value objectives robust by adding adversarial loss terms; or (b) employing “maximin” (i.e., maximizing the minimum value) notions of robustness. While regularization approaches are adept at reducing the probability of successful attacks, their performance drops significantly when an attack is successful. On the other hand, maximin objectives, while robust, can be extremely conservative. To this end, we focus on optimizing a well-studied robustness objective, namely regret. To ensure the solutions provided are not too conservative, we optimize an approximation of regret using three different methods. We demonstrate that our methods outperform existing best approaches for adversarial RL problems across a variety of standard benchmarks from literature.



This document contains a reference implementation for our CCER method in continuous action spaces. Our code is based on WocaR-PPO (Liang et al. 2022) codebase: [WocaR](https://github.com/umd-huang-lab/WocaR-RL/)


## 1. Requirements/Installation

Please run the following command to install required packages (suggested python version: 3.7.0)

```
# requirements
pip install -r requirements.txt

# need auto_LiRPA (Kaidi Xu, et al. 2020)
git clone https://github.com/KaidiXu/auto_LiRPA
cd auto_LiRPA
python setup.py install

# enter the trainer document
cd ../ccer
```

The commands provide default hyperparameters.

### PPO
- Use the following commands to train CCER:

```
python run.py --config-path configs/config_halfcheetah_robust_q_ppo_sgld.json --mode ccer
```

This will save an experiment folder at halfcheetah_robust_q_sgld/agents/YOUR_EXP_ID, where YOUR_EXP_ID is a randomly generated experiment ID (or, can be specified with '''--exp-id <id>'''). You can extract the best model from this folder by running:

```
python get_best_pickle.py halfcheetah_robust_q_sgld/agents/YOUR_EXP_ID
```
which will generate an adversary model best_model.YOUR_EXP_ID.model, for example best_model.7d48fb45.model.

To train for different MuJoCo environment, simply change the config_path and load_model in the command above and switch epsilon to the value that we report in our paper. 

- Evaluating:

```
python test.py --config-path configs/config_halfcheetah_robust_q_ppo_sgld.json --load-model models/PPO/model-ppo-cheetah.model --deterministic --attack-method action 
```
```
python test.py --config-path configs/config_halfcheetah_robust_q_ppo_sgld.json --exp-id YOUR_EXP_ID --row -1 --deterministic --attack-method action 
```

