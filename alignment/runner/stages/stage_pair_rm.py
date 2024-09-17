# -*- coding: utf-8 -*-
# @Time    : 5/31/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : stage_pair_rm.py
import copy
import itertools
import logging
import os
from typing import Dict, Any

import datasets
import llm_blender
import ray
import torch
from hydra.utils import instantiate
from mm_video.config import runner_store
from mm_video.utils.common import chunk
from omegaconf import DictConfig

from alignment.runner.stages.base import BaseStage
from alignment.runner.utils.common import save_sharded_dataset, retry_load_sharded_dataset

logger = logging.getLogger(__name__)

__all__ = ["PairRMStage", "get_preference_pair"]


def get_preference_pair(example: Dict[str, Any], add_system_message: bool):
    """
    Build preference pair based on the rank given by PairRM
    :param example:
    :param add_system_message: Add an empty system message
    :return:
    """
    prompt = example["prompt"]
    responses = example["responses"]
    ranks = example["ranks"]

    min_idx = min((i for i, rank in enumerate(ranks) if rank is not None), default=None, key=ranks.__getitem__)
    max_idx = max((i for i, rank in enumerate(ranks) if rank is not None), default=None, key=ranks.__getitem__)

    if min_idx is None or max_idx is None or ranks[max_idx] - ranks[min_idx] < 1:
        return {"prompt": None, "chosen": None, "rejected": None}

    messages = []

    # Process system message
    if "messages" in example and example["messages"][0]["role"] == "system":
        system_message = example["messages"][0]["content"]
    else:
        system_message = None
    if add_system_message:
        system_message = ""
    # Add system message
    if system_message is not None:
        messages.append({"role": "system", "content": system_message})

    # Add current user prompt
    messages.append({"role": "user", "content": prompt})

    return {
        "prompt": prompt,
        "chosen": copy.deepcopy(messages) + [{"role": "assistant", "content": responses[min_idx]}],
        "rejected": copy.deepcopy(messages) + [{"role": "assistant", "content": responses[max_idx]}]
    }


@runner_store(stage="pair_rm", output_dir="${hydra:runtime.output_dir}")
class PairRMStage(BaseStage):
    def __init__(
            self,
            iteration: int,
            stage: str,
            output_dir: str,
            resume: bool = False,
            add_system_message: bool = False,
            shard_id: int = 0,
            num_shards: int = 1,
            batch_size: int = 1,
            n_instances_per_gpu: int = 1
    ):
        super().__init__(iteration=iteration, stage=stage, output_dir=output_dir, resume=resume)
        self.n_instances_per_gpu = n_instances_per_gpu
        self.batch_size = batch_size

        @ray.remote(num_gpus=1 / self.n_instances_per_gpu)
        class BlenderActor:
            def __init__(self):
                self.blender = llm_blender.Blender()
                self.blender.loadranker("llm-blender/PairRM")

            def rank(self, *args):
                if len(args[1]) == 0:  # Responses are empty
                    return []
                return self.blender.rank(*args, disable_tqdm=False, batch_size=batch_size).tolist()

        self.actors = [BlenderActor.remote() for _ in range(torch.cuda.device_count() * self.n_instances_per_gpu)]
        self.actor_pool = ray.util.ActorPool(self.actors)

        self.add_system_message = add_system_message
        self.shard_id = shard_id
        self.num_shards = num_shards

    def get_pair_rm_rank(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """

        :param dataset:
        :return: Dataset, added with new column: ranks
        """
        prompts = dataset["prompt"]
        responses = dataset["responses"]

        chunked_prompts = chunk(prompts, n_chunks=self.n_instances_per_gpu * torch.cuda.device_count())
        chunked_responses = chunk(responses, n_chunks=self.n_instances_per_gpu * torch.cuda.device_count())
        assert len(chunked_prompts) == len(chunked_responses)

        ranks = list(itertools.chain.from_iterable(
            self.actor_pool.map(
                lambda a, v: a.rank.remote(*v),
                [(p, r) for p, r in zip(chunked_prompts, chunked_responses)]
            )
        ))
        assert len(ranks) == len(dataset), f"{len(ranks)} != {len(dataset)}"

        dataset = dataset.add_column(name="ranks", column=ranks)
        return dataset

    def apply_get_preference_pair_to_dataset(self, dataset_with_scores: datasets.Dataset) -> datasets.Dataset:
        return dataset_with_scores.map(
            get_preference_pair,
            num_proc=8,
            fn_kwargs={"add_system_message": self.add_system_message},
            desc="Getting preference pair",
            load_from_cache_file=False,
            keep_in_memory=True
        )

    def _run(self, cfg: DictConfig):
        prompt_dataset_with_resp: datasets.Dataset = instantiate(cfg.dataset)
        # Split into multiple shards
        prompt_dataset_with_resp = prompt_dataset_with_resp.shard(num_shards=self.num_shards, index=self.shard_id)
        # Ranking
        dataset_with_rank = self.get_pair_rm_rank(prompt_dataset_with_resp)
        save_sharded_dataset(
            dataset=dataset_with_rank,
            dataset_root=os.path.join(self.output_dir, "dataset_with_rank"),
            shard_id=self.shard_id, n_shards=self.num_shards
        )
        # Format data, remove unused columns
        preference_dataset = self.apply_get_preference_pair_to_dataset(dataset_with_rank)
        preference_dataset = preference_dataset.filter(
            lambda example: all(k in example and example[k] is not None for k in ["prompt", "chosen", "rejected"])
        )
        preference_dataset = preference_dataset.remove_columns(
            [x for x in list(preference_dataset.features) if x not in ["prompt", "chosen", "rejected"]]
        )
        save_sharded_dataset(
            dataset=preference_dataset,
            dataset_root=os.path.join(self.output_dir, "preference_dataset"),
            shard_id=self.shard_id, n_shards=self.num_shards
        )

    def resume_check(self):
        retry_load_sharded_dataset(os.path.join(self.output_dir, "preference_dataset"), retry=1, wait_interval=1)

    def is_rank_0(self) -> bool:
        return self.shard_id == 0
