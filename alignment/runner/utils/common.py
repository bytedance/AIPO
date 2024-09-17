# -*- coding: utf-8 -*-
# @Time    : 4/23/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : common.py

import logging
import os
import re
import time
from typing import Optional, Dict, List

import torch
from datasets import concatenate_datasets, load_from_disk, Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from alignment.chat_template import get_chat_template

__all__ = [
    "get_tokenizer", "update_torch_dtype",
    "save_sharded_dataset", "load_sharded_dataset", "retry_load_sharded_dataset",
    "retry_load_from_disk", "check_conversation"
]

logger = logging.getLogger(__name__)

DATASET_SHARD_PREFIX = "shard"


def get_tokenizer(
        model_name_or_path: str,
        chat_template: Optional[str] = None,
        revision: Optional[str] = None
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.model_max_length > 100_100:
        tokenizer.model_max_length = 2048
    if chat_template is not None:
        tokenizer.chat_template = get_chat_template(chat_template)
    tokenizer.truncation = True
    tokenizer.padding = "longest"
    return tokenizer


def update_torch_dtype(model_init_kwargs: dict):
    """
    Update the model_init_kwargs in-place, replace torch dtype.
    :param model_init_kwargs:
    :return:
    """
    if "torch_dtype" in model_init_kwargs:
        model_init_kwargs["torch_dtype"] = (
            model_init_kwargs["torch_dtype"] if model_init_kwargs["torch_dtype"] in ["auto", None]
            else getattr(torch, model_init_kwargs["torch_dtype"])
        )


def load_sharded_dataset(dataset_root: str) -> Dataset:
    """
    Load a sharded dataset.
    :param dataset_root: The root directory where the shards are saved.
    :return: The loaded dataset.
    """
    shard_files = os.listdir(dataset_root)
    shard_regex = re.compile(rf"{DATASET_SHARD_PREFIX}-(\d+)-of-(\d+)")
    shard_ids = []
    n_shards = None

    for shard_file in shard_files:
        match = shard_regex.match(shard_file)
        if match:
            shard_id = match.group(1)
            n_shards = match.group(2)
            shard_ids.append(shard_id)

    logger.debug("dataset root: %s", dataset_root)
    logger.debug("shard_ids: %s", shard_ids)
    logger.debug("n_shards: %s", n_shards)

    assert n_shards is not None, f"No shard files found in the dataset root directory: {dataset_root}"
    assert len(shard_ids) == int(n_shards), ("Missing shard files in the dataset root directory: "
                                             f"{shard_ids} vs {int(n_shards)}")

    shard_ids.sort(key=lambda x: int(x))

    datasets = []
    for shard_id in shard_ids:
        load_loc = os.path.join(dataset_root, f"{DATASET_SHARD_PREFIX}-{shard_id}-of-{n_shards}")
        dataset = load_from_disk(load_loc, keep_in_memory=True)
        datasets.append(dataset)

    loaded_dataset = concatenate_datasets(datasets)
    logger.debug("Load dataset:\n%s", loaded_dataset)
    return loaded_dataset


def retry_load_sharded_dataset(dataset_root: str, retry: int = 30, wait_interval: int = 60) -> Dataset:
    exception = None
    for i in range(retry):
        try:
            return load_sharded_dataset(dataset_root)
        except Exception as e:
            exception = e
            logger.warning("Retry %s, got exception: %s", i, exception)
        time.sleep(wait_interval)
    if exception is not None:
        logger.exception("Got exception after try for %s times: %s", retry, exception)
        raise exception


def save_sharded_dataset(dataset: Dataset, dataset_root: str, shard_id: int, n_shards: int) -> str:
    """
    Save dataset in shards.
    :param dataset:
    :param dataset_root:
    :param shard_id:
    :param n_shards:
    :return:
    """
    assert type(shard_id) is int, "shard_id must be an integer"
    assert type(n_shards) is int, "n_shards must be an integer"
    assert shard_id < n_shards, "shard_id must be less than n_shards"

    save_loc = os.path.join(dataset_root, f"{DATASET_SHARD_PREFIX}-{shard_id:05d}-of-{n_shards:05d}")
    logger.debug("Saving sharded dataset to %s", save_loc)
    dataset.save_to_disk(save_loc)
    return save_loc


def retry_load_from_disk(dataset_path: str, retry: int = 30, wait_interval: float = 60) -> Dataset:
    exception = None
    for i in range(retry):
        try:
            return load_from_disk(dataset_path, keep_in_memory=True)
        except Exception as e:
            exception = e
            logger.warning("Retry %s, got exception: %s", i, exception)
        time.sleep(wait_interval)
    if exception is not None:
        logger.exception("Got exception after try for %s times: %s", retry, exception)
        raise exception


def check_conversation(messages: List[Dict[str, str]]):
    assert all(x["role"] == "user" if i % 2 == 0 else x["role"] == "assistant" for i, x in enumerate(messages))
