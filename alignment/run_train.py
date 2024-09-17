# -*- coding: utf-8 -*-
# @Time    : 2024/2/27
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : run_train.py

import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

###########################
# Register, do not remove #
###########################
# noinspection PyUnresolvedReferences
import alignment.dataset
# noinspection PyUnresolvedReferences
import alignment.runner.stages
# noinspection PyUnresolvedReferences
import alignment.utils.distributed_resolver
# noinspection PyUnresolvedReferences
import alignment.utils.iteration_resolver

__all__ = ["main"]


@hydra.main(version_base=None,
            config_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs"),
            config_name="config")
def main(cfg: DictConfig) -> None:
    runner = instantiate(cfg.runner)
    runner.run(cfg)


if __name__ == '__main__':
    main()
