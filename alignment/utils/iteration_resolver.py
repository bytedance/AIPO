# -*- coding: utf-8 -*-
# @Time    : 7/20/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : iteration_resolver.py

from omegaconf import OmegaConf, DictConfig


def last_iteration(_root_: DictConfig) -> int:
    return _root_.runner.iteration - 1


def current_iteration(_root_: DictConfig) -> int:
    return _root_.runner.iteration


OmegaConf.register_new_resolver("iteration.last", last_iteration)
OmegaConf.register_new_resolver("iteration.current", current_iteration)
