# PIMbot

Official implementations and results of IROS 2023 paper [PIMbot: Policy and Incentive Manipulation for Multi-Robot Reinforcement Learning in Social Dilemmas](https://arxiv.org/pdf/2307.15944).

## Overview
PIMbot introduces two forms of reward function manipulation in multi-agent reinforcement learning (RL) social dilemmas:

1.	Policy Manipulation: Adjusting the decision-making strategies of robots to influence task outcomes.

2.	Incentive Manipulation: Modifying reward structures to change robot behavior within a social dilemma.

## Setup

- Goto ./lio folder.
- Python 3.6
- Tensorflow >= 1.12
- OpenAI Gym == 0.10.9
- Clone and `pip install` [Sequential Social Dilemma](https://github.com/011235813/sequential_social_dilemma_games), which is a fork from the [original](https://github.com/eugenevinitsky/sequential_social_dilemma_games) open-source implementation.
- Clone and `pip install` [LOLA](https://github.com/alshedivat/lola) if you wish to run this baseline.
- Clone this repository and run `$ pip install -e .` from the root.

## Navigation
* `./*.ipynb` - Plot and visualization scripts.
* `./*.py` - Plot scripts.
* `./*.png` - Experimental results in PIMbot paper.
* `./lio/alg/` - Implementation of LIO and PG/AC baselines
* `./lio/env/` - Implementation of the Escape Room game and wrappers around the SSD environment.
* `./lio/results/` - Results of training will be stored in subfolders here. Each independent training run will create a subfolder that contains the final Tensorflow model, and reward log files. For example, 5 parallel independent training runs would create `results/cleanup/10x10_lio_0`,...,`results/cleanup/10x10_lio_4` (depending on configurable strings in config files).
* `./lio/utils/` - Utility methods.

## Examples

### Train LIO on Escape Room

* Set config values in `alg/config_room_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio er`. Default settings conduct 5 parallel runs with different seeds.
* For a single run, execute `$ python train_lio.py er`.

### Train LIO on Cleanup

* Set config values in `alg/config_ssd_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio ssd`.
* For a single run, execute `$ python train_ssd.py`.

## Citation

Please cite our paper if you are inspired by PIMbot in your work:

<pre>
@inproceedings{nikkhoo2023pimbot,
  title={Pimbot: Policy and incentive manipulation for multi-robot reinforcement learning in social dilemmas},
  author={Nikkhoo, Shahab and Li, Zexin and Samanta, Aritra and Li, Yufei and Liu, Cong},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={5630--5636},
  year={2023},
  organization={IEEE}
}
</pre>

## Acknowledgement

Code is implemented based on [Learning to Incentivize Other Learning Agents](https://github.com/011235813/lio). We would like to thank the authors for making their code public.

## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
