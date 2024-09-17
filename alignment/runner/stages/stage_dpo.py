# -*- coding: utf-8 -*-
# @Time    : 5/22/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : stage_dpo.py
import copy
import logging
import os
from dataclasses import field
from typing import Optional, Dict, Any

import datasets
import torch.distributed as dist
from hydra.utils import instantiate
from mm_video.config import runner_store
from mm_video.utils.common import get_rank
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from alignment.runner.stages.base import TrainerBaseStage, ModelConfig
from alignment.runner.utils.common import check_conversation
from alignment.trainer.customized_dpo_trainer import (
    CustomizedDPOTrainer,
    CustomizedDPOConfig,
    HydraCompatCustomizedDPOConfig,
)

logger = logging.getLogger(__name__)


def apply_chat_template(example: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    """
    Apply chat template for preference dataset.
    Dataset format: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
    :param example:
    :param tokenizer: Huggingface tokenizer
    :return:
    """
    assert example["chosen"][0]["role"] == example["rejected"][0]["role"]
    if example["chosen"][0]["role"] == "system":
        assert example["chosen"][0]["content"] == example["rejected"][0]["content"]
        system_message = example["chosen"][0]
        assert example["rejected"][0]["role"] == "system"
        chosen_messages = example["chosen"][1:]
        rejected_messages = example["rejected"][1:]
    else:
        system_message = None
        chosen_messages = example["chosen"]
        rejected_messages = example["rejected"]

    check_conversation(chosen_messages)
    check_conversation(rejected_messages)

    if system_message:
        prompt_messages = [system_message]
    else:
        prompt_messages = []
    assert chosen_messages[0]["content"] == rejected_messages[0]["content"]
    prompt_messages.append(copy.deepcopy(chosen_messages[0]))

    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=False),
        "chosen": chosen_messages[1]["content"],
        "rejected": rejected_messages[1]["content"]
    }


def filter_truncation(
        example: Dict[str, Any], tokenizer: PreTrainedTokenizer,
        max_prompt_length: int, max_response_length: int
) -> bool:
    """
    Filter dataset to avoid truncation of prompts and responses.
    :param example:
    :param tokenizer:
    :param max_prompt_length:
    :param max_response_length:
    :return:
    """
    prompt_length = len(tokenizer.tokenize(example["prompt"]))
    if prompt_length > max_prompt_length:
        return False
    chosen_length = len(tokenizer.tokenize(example["chosen"]))
    if chosen_length > max_response_length:  # Test immediately to reduce computation.
        return False
    rejected_length = len(tokenizer.tokenize(example["rejected"]))
    if rejected_length > max_response_length:
        return False
    return True


# noinspection HttpUrlsUsage
def filter_urls(
        example: Dict[str, Any], ignore_rejected: bool = False
) -> bool:
    """
    Filter examples with URLs.
    :param example:
    :param ignore_rejected:
    :return:
    """
    if "http://" in example["chosen"] or "https://" in example["chosen"]:
        return False
    if not ignore_rejected and ("http://" in example["rejected"] or "https://" in example["rejected"]):
        return False
    return True


@runner_store(stage="dpo", output_dir="${hydra:runtime.output_dir}", training_args=HydraCompatCustomizedDPOConfig)
class DPOStage(TrainerBaseStage):
    def __init__(
            self,
            iteration: int,
            stage: str,
            output_dir: Optional[str],
            resume: bool,
            # Config nodes
            model: ModelConfig = field(default_factory=ModelConfig),
            training_args: CustomizedDPOConfig = field(default_factory=CustomizedDPOConfig),
            # Additional config
            filter_urls: bool = False,
            ignore_rejected: bool = False,
    ):
        super().__init__(
            iteration=iteration,
            stage=stage,
            output_dir=output_dir,
            model=model,
            training_args=training_args,
            resume=resume
        )
        self.training_args: CustomizedDPOConfig
        self.filter_urls = filter_urls
        self.ignore_rejected = ignore_rejected

    def dpo(self, dpo_dataset: datasets.Dataset):
        trainer = CustomizedDPOTrainer(
            model=self.model_name,
            ref_model=self.model_name,
            args=self.training_args,
            train_dataset=dpo_dataset,
            tokenizer=self.tokenizer
        )

        logger.info("Training...")
        trainer.train()

        logger.info("Saving model...")
        logger.info("Model will be saved to %s", self.training_args.output_dir)
        trainer.save_model()
        # noinspection PyArgumentList
        trainer.save_state()

        return trainer

    def _run(self, cfg: DictConfig):
        logger.info("Loading preference dataset...")
        preference_dataset: datasets.Dataset = instantiate(cfg.dataset)
        logger.info("Preference dataset is loaded.")

        logger.info("Applying chat template to preference dataset...")
        preference_dataset = preference_dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer},
            num_proc=16, desc="Apply chat template",
            load_from_cache_file=False, keep_in_memory=True
        )
        logger.info("Chat template is applied successfully.")

        # Only keep the required columns
        preference_dataset = preference_dataset.select_columns(["prompt", "chosen", "rejected"])
        # Save dataset for debugging purpose
        if get_rank() == 0:
            preference_dataset.save_to_disk(os.path.join(self.output_dir, "train_dataset"))

        # Apply filters
        logger.info("Applying filtering to preference dataset...")
        logger.info(">> Filter: truncation")
        preference_dataset = preference_dataset.filter(
            filter_truncation,
            fn_kwargs=dict(
                tokenizer=self.tokenizer, max_prompt_length=self.training_args.max_prompt_length,
                max_response_length=self.training_args.max_length - self.training_args.max_prompt_length - 32
            ),
            num_proc=16, desc="Filtering truncation", load_from_cache_file=False, keep_in_memory=True
        )
        if get_rank() == 0:
            preference_dataset.save_to_disk(os.path.join(self.output_dir, "train_dataset_remove_trunc"))
        if self.filter_urls:
            logger.info(">> Filter: URLs")
            preference_dataset = preference_dataset.filter(
                filter_urls,
                fn_kwargs=dict(ignore_rejected=self.ignore_rejected),
                num_proc=16, desc="Filtering URLs", load_from_cache_file=False, keep_in_memory=True
            )
            if get_rank() == 0:
                preference_dataset.save_to_disk(os.path.join(self.output_dir, "train_dataset_filter_urls"))
        logger.info("Filtering is applied successfully.")

        self.dpo(dpo_dataset=preference_dataset)

        dist.barrier()
        logger.info("Training completed.")
