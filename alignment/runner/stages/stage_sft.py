# -*- coding: utf-8 -*-
# @Time    : 5/22/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : ${FILE_NAME}
import logging
from dataclasses import field
from typing import Optional

import datasets
from hydra.utils import instantiate
from hydra_zen import builds, just
from mm_video.config import runner_store
from omegaconf import DictConfig
from trl import SFTTrainer, SFTConfig

from alignment.runner.stages.base import TrainerBaseStage, ModelConfig

logger = logging.getLogger(__name__)

CompatSFTConfig = builds(
    SFTConfig,
    output_dir="${hydra:runtime.output_dir}",
    deepspeed=just(dict()),
    populate_full_signature=True,
    hydra_convert="all"
)


@runner_store(stage="sft", output_dir="${hydra:runtime.output_dir}", training_args=CompatSFTConfig)
class SFTStage(TrainerBaseStage):
    def __init__(
            self,
            iteration: int,
            stage: str,
            output_dir: Optional[str],
            resume: bool = False,
            # Config nodes
            model: ModelConfig = field(default=ModelConfig),
            training_args: SFTConfig = field(default_factory=SFTConfig),
    ):
        super().__init__(
            iteration=iteration,
            stage=stage,
            output_dir=output_dir,
            resume=resume,
            model=model,
            training_args=training_args,
        )

    def supervised_finetuning(self, sft_dataset: datasets.Dataset):
        logger.info(">>> Fine-tuning with SFT trainer...")
        trainer = SFTTrainer(
            model=self.model_name,
            model_init_kwargs=self.model_init_kwargs,
            args=self.training_args,
            train_dataset=sft_dataset,
            tokenizer=self.tokenizer,
            max_seq_length=2048,
            packing=True
        )
        trainer.train()
        trainer.save_model()
        return trainer

    def _run(self, cfg: DictConfig):
        sft_dataset: datasets.Dataset = instantiate(cfg.dataset)
        assert "messages" in list(sft_dataset.features)

        self.supervised_finetuning(sft_dataset=sft_dataset)
