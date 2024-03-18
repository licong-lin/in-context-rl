# Transformers as Decision Makers

This repository contains the code for the experiments in the paper "Transformers as Decision Makers: Provable In-Context Reinforcement Learning via Supervised Pretraining". 



### Abstract
>Large transformer models pretrained on offline reinforcement learning datasets have demonstrated remarkable in-context reinforcement learning (ICRL) capabilities, where they can make good decisions when prompted with interaction trajectories from unseen environments. However, when and how transformers can be trained to perform ICRL have not been theoretically well-understood. In particular, it is unclear which reinforcement-learning algorithms transformers can perform in context, and how distribution mismatch in offline training data affects the learned algorithms. This paper provides a theoretical framework that analyzes supervised pretraining for ICRL. This includes two recently proposed training methods -- algorithm distillation and decision-pretrained transformers. First, assuming model realizability, we prove the supervised-pretrained transformer will imitate the conditional expectation of the expert algorithm given the observed trajectory. The generalization error will scale with model capacity and a distribution divergence factor between the expert and offline algorithms. Second, we show transformers with ReLU attention can efficiently approximate near-optimal online reinforcement learning algorithms like LinUCB and Thompson sampling for stochastic linear bandits, and UCB-VI for tabular Markov decision processes. This provides the first quantitative analysis of the ICRL capabilities of transformers pretrained from offline trajectories.


## Miscellanous

The code is built upon an early version of https://github.com/jon--lee/decision-pretrained-transformer.  
More information about the code can also be found in the above repo.

If you use this code in your research, please cite our paper

```bibtex
@article{lin2023transformers,
  title={Transformers as decision makers: Provable in-context reinforcement learning via supervised pretraining},
  author={Lin, Licong and Bai, Yu and Mei, Song},
  journal={arXiv preprint arXiv:2310.08566},
  year={2023}
}
