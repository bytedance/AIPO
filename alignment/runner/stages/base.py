# -*- coding: utf-8 -*-
# @Time    : 5/22/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : base.py
import copy
import datetime
import functools
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing import Optional, Any, Union

import torch.distributed as dist
from hydra_zen import builds, just
from mm_video.utils.language import VLLMGenerator, HFPipelineGenerator
from omegaconf import DictConfig, MISSING
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from alignment.runner.utils.common import update_torch_dtype, get_tokenizer

logger = logging.getLogger(__name__)

# Making hf config compatible with hydra
HydraCompatTrainingArguments = builds(
    TrainingArguments,
    output_dir="${hydra:runtime.output_dir}",
    deepspeed=just(dict()),  # Including deepspeed config in yaml config of Hydra
    populate_full_signature=True,
    hydra_convert="all"
)


@dataclass
class ModelConfig:
    model_name: str = MISSING
    model_init_kwargs: dict = field(default_factory=dict)
    chat_template: Optional[str] = None


@dataclass
class GenerateConfig:
    max_new_tokens: int = 512
    # HF Generator
    batch_size: int = 8
    # vLLM Generator
    vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.9
    tensor_parallel: int = 1
    vllm_model_max_length: Optional[int] = None
    vllm_enable_chunked_prefill: bool = False  # Note: not compatible with sliding window
    vllm_max_num_batched_tokens: Optional[int] = None
    # Diversity
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    n_candidates: int = 4
    n_sampled_decoding: int = 3
    # Sharding
    shard_id: int = 0
    num_shards: int = 1


class BaseStage(ABC):
    def __init__(self, iteration: int, stage: str, output_dir: Optional[str] = None, resume: bool = False):
        logger.info("##################%s#########%s#######", "#" * len(str(iteration)), "#" * len(str(stage)))
        logger.info("###### Iteration: %s, Stage: %s ######", iteration, stage)
        logger.info("##################%s#########%s#######", "#" * len(str(iteration)), "#" * len(str(stage)))
        self.iteration = iteration
        self.stage = stage
        self.output_dir = output_dir
        self.resume = resume

    # noinspection PyMethodMayBeStatic
    def resume_check(self):
        """
        Return True if it is possible to resume from the previous training progress.
        This function will return False by default (do not resume from previous training progress).
        One should override this method if you want to resume from a previous training progress.
        :return:
        """
        raise NotImplementedError("Resume logic is not implemented.")

    def skip(self) -> bool:
        # Always return False if resume is not enabled
        if not self.resume:
            return False

        if not os.path.exists(self.output_dir):
            logger.debug("Output directory does not exist, will not check for resume.")
            return False

        try:
            logger.info("Checking if possible to resume from previous training progress.")
            self.resume_check()
            logger.info("Resume check passed.")
            return True
        except Exception as e:
            logger.info("Cannot resume due to the following reason: %s", str(e))
            return False

    def cleanup(self):
        if not os.path.exists(self.output_dir):
            return

        backup_prefix = "backup"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(self.output_dir, f'{backup_prefix}_{timestamp}')

        exclude = [rf"^{backup_prefix}.*", r".*.log$", r"^\..*"]

        try:
            os.makedirs(backup_dir, exist_ok=True)
            logger.info("Backup directory created: %s", backup_dir)

            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)

                # Move each file to the backup directory
                if any(re.match(r, filename) is not None for r in exclude):
                    logger.debug("Exclude: %s", file_path)
                elif os.path.isfile(file_path):
                    shutil.move(file_path, os.path.join(backup_dir, filename))
                    logger.debug("Backed up and deleted file: %s", file_path)
                elif os.path.isdir(file_path):
                    shutil.move(file_path, os.path.join(backup_dir, filename))
                    logger.debug("Backed up and deleted directory: %s", file_path)

            logger.info("Cleanup complete.")
        except Exception as e:
            logger.error("Error during cleanup: %s", e)

    @abstractmethod
    def is_rank_0(self) -> bool:
        pass

    def run(self, cfg: DictConfig):
        """
        Call `self.run` conditionally
        :param cfg:
        :return:
        """
        if self.skip():
            logger.info("Skipping %s stage", self.stage)
            return
        else:
            if self.is_rank_0():
                self.cleanup()
            self._run(cfg)

    @abstractmethod
    def _run(self, cfg: DictConfig):
        pass


class GenerateBaseStage(BaseStage, ABC):
    """
    Basic template for generating with a generator, containing a generate config
    """

    def __init__(
            self,
            iteration: int,
            stage: str,
            output_dir: Optional[str] = None,
            resume: bool = False,
            # Config nodes
            model: ModelConfig = field(default=ModelConfig),
            generate: GenerateConfig = field(default=GenerateConfig),
    ):
        super().__init__(iteration=iteration, stage=stage, output_dir=output_dir, resume=resume)

        self.model_cfg = model
        self.generate_cfg = generate

        self.model_name = copy.deepcopy(model.model_name)
        self.model_init_kwargs = copy.deepcopy(model.model_init_kwargs)
        update_torch_dtype(self.model_init_kwargs)

        self.tokenizer = get_tokenizer(
            model_name_or_path=self.model_name,
            chat_template=self.model_cfg.chat_template,
            revision=self.model_init_kwargs["revision"] if "revision" in self.model_init_kwargs else None
        )

    @functools.lru_cache()
    def _build_generator(self) -> Union[VLLMGenerator, HFPipelineGenerator]:
        if self.generate_cfg.vllm:
            return VLLMGenerator(
                model_name_or_path=self.model_name,
                model_init_kwargs=dict(
                    gpu_memory_utilization=self.generate_cfg.vllm_gpu_memory_utilization,
                    swap_space=64,
                    dtype="bfloat16",
                    tensor_parallel_size=self.generate_cfg.tensor_parallel,
                    max_model_len=self.generate_cfg.vllm_model_max_length,
                    enable_chunked_prefill=self.generate_cfg.vllm_enable_chunked_prefill,
                    max_num_batched_tokens=self.generate_cfg.vllm_max_num_batched_tokens,
                    tokenizer_pool_size=1
                ),
                sampling_params_kwargs=dict(
                    max_tokens=self.generate_cfg.max_new_tokens,
                    temperature=self.generate_cfg.temperature,
                    top_p=self.generate_cfg.top_p,
                    top_k=self.generate_cfg.top_k
                )
            )
        else:
            return HFPipelineGenerator(
                self.model_name, self.model_init_kwargs, self.tokenizer,
                batch_size=self.generate_cfg.batch_size,
                generate_kwargs=dict(
                    max_new_tokens=self.generate_cfg.max_new_tokens,
                    # Set to True by default to get diverse output.
                    # Also required by `num_return_sequences>1` to avoid greedy decoding.
                    do_sample=True,
                    temperature=self.generate_cfg.temperature,
                    top_p=self.generate_cfg.top_p,
                    top_k=self.generate_cfg.top_k,
                ),
                pipeline_kwargs=dict(
                    device="cuda"
                )
            )

    def is_rank_0(self) -> bool:
        return self.generate_cfg.shard_id == 0


class TrainerBaseStage(BaseStage, ABC):
    """
    Basic template for training with huggingface trainer
    """

    def __init__(
            self,
            iteration: int,
            stage: str,
            output_dir: Optional[str],
            resume: bool = False,
            # Config nodes
            model: ModelConfig = field(default=ModelConfig),
            training_args: Any = None,
    ):
        super().__init__(iteration=iteration, stage=stage, output_dir=output_dir, resume=resume)

        self.model_cfg = model
        self.training_args = training_args

        self.model_name = copy.deepcopy(model.model_name)
        self.model_init_kwargs = copy.deepcopy(model.model_init_kwargs)
        # update_torch_dtype(self.model_init_kwargs) # Should not use for trl>=0.9.0

        self.tokenizer = get_tokenizer(
            model_name_or_path=self.model_name,
            chat_template=self.model_cfg.chat_template,
            revision=self.model_init_kwargs["revision"] if "revision" in self.model_init_kwargs else None
        )

    def resume_check(self):
        AutoModelForCausalLM.from_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.output_dir)
        if dist.is_initialized():
            dist.barrier()

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0
