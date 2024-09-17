# -*- coding: utf-8 -*-
# @Time    : 2024/2/5
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : __init__.py

import warnings

warnings.warn("You are importing alignment.dataset, this will import submodules recursively to register config to "
              "Hydra.", UserWarning)

from . import chat_dataset
from . import hf_dataset
