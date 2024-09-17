# -*- coding: utf-8 -*-
# @Time    : 5/22/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : stage_barrier.py

"""
`BarrierStage` is used to ensure synchronization in multi-node training.
"""

import logging
import time

from mm_video.config import runner_store
from omegaconf import DictConfig
from torch import distributed as dist

from alignment.runner.stages.base import BaseStage

logger = logging.getLogger(__name__)


@runner_store(stage="barrier")
class BarrierStage(BaseStage):
    def __init__(self, iteration: int, stage: str, init_process_group: bool = False, sleep: int = 180):
        super().__init__(iteration=iteration, stage=stage)
        self.init_process_group = init_process_group
        self.sleep = sleep

    def _run(self, cfg: DictConfig):
        logger.info("Entered barrier at %s", time.ctime())
        if self.init_process_group:
            dist.init_process_group(backend="nccl")
            logger.info("Process group is initialized successfully.")
        else:
            logger.info("Init process group is not disabled.")
        time.sleep(self.sleep)
        logger.info("Exiting barrier at %s", time.ctime())

    def is_rank_0(self) -> bool:
        return False
