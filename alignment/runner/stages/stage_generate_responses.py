# -*- coding: utf-8 -*-
# @Time    : 5/22/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : stage_generate_responses.py
import logging
import os
from dataclasses import field
from typing import Optional, Dict, Any

import datasets
from hydra.utils import instantiate
from mm_video.config import runner_store
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from alignment.runner.stages.base import GenerateBaseStage, ModelConfig, GenerateConfig
from alignment.runner.utils.common import save_sharded_dataset, retry_load_sharded_dataset

logger = logging.getLogger(__name__)

__all__ = [
    "apply_chat_template", "GenerateResponsesStage"
]


def filter_prompt(
        example: Dict[str, Any], tokenizer: PreTrainedTokenizer,
        prompt_min_length: int, prompt_max_length: int
) -> bool:
    prompt_length = len(tokenizer.tokenize(example["prompt"], add_special_tokens=False))
    if prompt_length < prompt_min_length or prompt_length > prompt_max_length:
        return False
    return True


def apply_chat_template(example: Dict[str, Any], tokenizer: PreTrainedTokenizer, add_system_message: bool):
    if example["messages"][0]["role"] != "system":
        messages = example["messages"]
        prompt_messages = [messages[0]]
        if add_system_message:
            prompt_messages.insert(0, {"role": "system", "content": ""})
    else:
        messages = example["messages"][1:]
        prompt_messages = [example["messages"][0], messages[0]]

    assert all(x["role"] == "user" if i % 2 == 0 else x["role"] == "assistant" for i, x in enumerate(messages))

    ret = {
        "prompt": messages[0]["content"],
        "prompt_with_chat_template": tokenizer.apply_chat_template(
            prompt_messages, tokenize=False,
            add_generation_prompt=True
        )
    }
    if len(messages) > 1:
        ret["gt"] = messages[1]["content"]
    return ret


@runner_store(stage="generate_responses", output_dir="${hydra:runtime.output_dir}")
class GenerateResponsesStage(GenerateBaseStage):
    def __init__(
            self,
            iteration: int,
            stage: str,
            output_dir: Optional[str],
            add_system_message: bool = False,
            include_gt: bool = False,  # Include ground truth in responses
            resume: bool = False,
            # Config nodes
            model: ModelConfig = field(default=ModelConfig),
            generate: GenerateConfig = field(default=GenerateConfig),
            log_prompt: bool = False,
            add_zero_temperature: bool = False,
            additional_temperature: Optional[float] = None,
            prompt_max_length: int = 512,
            prompt_min_length: int = 3,
    ):
        super().__init__(
            iteration=iteration,
            stage=stage,
            output_dir=output_dir,
            model=model,
            generate=generate,
            resume=resume
        )
        self.include_gt = include_gt
        self.add_system_message = add_system_message
        self.log_prompt = log_prompt
        self.add_zero_temperature = add_zero_temperature
        self.additional_temperature = additional_temperature
        self.prompt_max_length = prompt_max_length
        self.prompt_min_length = prompt_min_length

    def apply_chat_template_to_dataset(self, chat_dataset: datasets.Dataset) -> datasets.Dataset:
        return chat_dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": self.tokenizer, "add_system_message": self.add_system_message},
            num_proc=16,
            desc="Applying Chat Template",
            load_from_cache_file=False,
            keep_in_memory=True
        )

    def generate_candidate_responses(self, prompt_dataset: datasets.Dataset):
        generator = self._build_generator()

        # IMPORTANT: Use prompt with chat template as inputs
        assert "prompt_with_chat_template" in list(prompt_dataset.features)
        prompts = prompt_dataset[:]["prompt_with_chat_template"]

        # Generate, return N output for each prompt
        if self.generate_cfg.vllm:
            model_responses = generator.generate(
                prompts,
                n=self.generate_cfg.n_candidates
            )
        else:
            model_responses = generator.generate(
                prompts,
                num_return_sequences=self.generate_cfg.n_candidates,
            )
        model_responses = [[r.replace("</s>", "").lstrip() for r in group_resp] for group_resp in model_responses]

        # Add gt to the responses
        if self.include_gt:
            assert "gt" in list(prompt_dataset.features), "`include_gt=True` is set, but gt features missing"
            gt_answers = prompt_dataset[:]["gt"]
            for resp_group, gt in zip(model_responses, gt_answers):
                resp_group.append(gt)

        # Add the responses with temperature==0
        if self.add_zero_temperature:
            model_responses_zero_temperature = generator.generate(
                prompts,
                temperature=0.0
            )
            model_responses_zero_temperature = [
                r.replace("</s>", "").lstrip() for r in model_responses_zero_temperature
            ]
            assert len(model_responses_zero_temperature) == len(prompt_dataset)
            for resp_group, resp_zero_temp in zip(model_responses, model_responses_zero_temperature):
                resp_group.append(resp_zero_temp)

        # Add the responses with different temperature
        if self.additional_temperature is not None:
            if self.generate_cfg.vllm:
                model_responses_additional_temperature = generator.generate(
                    prompts,
                    n=self.generate_cfg.n_candidates,
                    temperature=self.additional_temperature
                )
            else:
                model_responses_additional_temperature = generator.generate(
                    prompts,
                    num_return_sequences=self.generate_cfg.n_candidates,
                    temperature=self.additional_temperature
                )
            model_responses_additional_temperature = [
                [r.replace("</s>", "").lstrip() for r in group_resp]
                for group_resp in model_responses_additional_temperature
            ]
            assert len(model_responses_additional_temperature) == len(prompt_dataset)
            for resp_group, resp_group_additional_temp in zip(model_responses, model_responses_additional_temperature):
                resp_group.extend(resp_group_additional_temp)

        assert len(model_responses) == len(prompt_dataset), f"{len(model_responses)} vs {len(prompt_dataset)}"

        prompt_dataset = prompt_dataset.add_column(name="responses", column=model_responses)
        return prompt_dataset

    def _run(self, cfg: DictConfig):
        # Initialize from chat dataset
        logger.info("Loading dataset...")
        prompt_dataset = instantiate(cfg.dataset)

        # shard before generating
        logger.info("Sharding: %s / %s", self.generate_cfg.shard_id, self.generate_cfg.num_shards)
        prompt_dataset_sharded = prompt_dataset.shard(
            num_shards=self.generate_cfg.num_shards,
            index=self.generate_cfg.shard_id
        )

        # filtering
        prompt_dataset_sharded = prompt_dataset_sharded.filter(
            filter_prompt,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "prompt_min_length": self.prompt_min_length,
                "prompt_max_length": self.prompt_max_length
            },
            num_proc=16, desc="Filtering prompts", load_from_cache_file=False, keep_in_memory=True
        )

        logger.info("Applying chat template to sharded dataset...")
        prompt_dataset_sharded = self.apply_chat_template_to_dataset(prompt_dataset_sharded)
        if self.log_prompt:
            logger.info("Saving sharded prompt dataset (with chat template applied)...")
            save_sharded_dataset(
                prompt_dataset_sharded,
                os.path.join(self.output_dir, "prompt_dataset"),
                shard_id=self.generate_cfg.shard_id, n_shards=self.generate_cfg.num_shards
            )

        logger.info("Generating candidate responses...")
        prompt_dataset_with_resp = self.generate_candidate_responses(prompt_dataset_sharded)

        logger.info("Saving results...")
        save_sharded_dataset(
            dataset=prompt_dataset_with_resp,
            dataset_root=os.path.join(self.output_dir, "prompt_dataset_with_resp"),
            shard_id=self.generate_cfg.shard_id, n_shards=self.generate_cfg.num_shards
        )

        logger.info("All done!")

    def resume_check(self):
        retry_load_sharded_dataset(os.path.join(self.output_dir, "prompt_dataset_with_resp"), retry=1, wait_interval=1)
