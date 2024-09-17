# -*- coding: utf-8 -*-
# @Time    : 5/22/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : __init__.py

import warnings

warnings.warn(
    "You are importing alignment.runner.stages, this will import submodules recursively to register config to Hydra.",
    UserWarning
)

from . import (
    stage_barrier,
    stage_dpo,
    stage_generate_responses,
    stage_self_instruct,
    stage_sft,
    stage_pair_rm,
)
