# -*- coding: utf-8 -*-
# @Time    : 2024/3/21
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : hf_dataset.py

"""
Methods for loading huggingface dataset, registered to Hydra config.
"""

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union

from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk
from mm_video.config import dataset_store

from alignment.runner.utils.common import retry_load_sharded_dataset, retry_load_from_disk

dataset_store(load_dataset, name="HFLoadDataset")

dataset_store(load_from_disk, name="HFLoadFromDisk")

dataset_store(retry_load_from_disk, name="HFRetryLoadFromDisk")

dataset_store(retry_load_sharded_dataset, name="LoadShardedDataset")


@dataset_store(name="ConcatDataset")
def load_concat_datasets(datasets: Dict[str, Dataset]):
    """
    Concatenates multiple datasets into a single dataset.

    :param datasets: A dictionary of datasets to concatenate, where keys are dataset names and values are Dataset
    objects.
    :return: A single concatenated Dataset object containing all the samples from the input datasets.
    """
    return concatenate_datasets(list(datasets.values()))


@dataset_store(name="LoadUltraFeedbackBinarized")
def load_ultra_feedback_binarized(
        n_samples: Union[int, None] = None,
        split: str = "train_prefs",
        seed: int = 0
) -> Dataset:
    """
    Loads the UltraFeedback Binarized dataset, optionally shuffling and selecting a subset of samples.

    :param n_samples: Optional; the number of samples to select from the dataset. If None, all samples are used.
    :param split: The dataset split to load, e.g., "train_prefs". Default is "train_prefs".
    :param seed: The seed for shuffling the dataset. Default is 0.
    :return: A HuggingFace Dataset object containing the selected subset of the UltraFeedback Binarized dataset.
    """
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    if n_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    return dataset


@dataset_store(name="LoadUltraFeedback")
def load_ultra_feedback(
        n_samples: Union[int, None] = None,
        seed: int = 0
) -> Dataset:
    """
    Loads the UltraFeedback dataset, optionally shuffling and selecting a subset of samples.
    Maps the dataset to include only 'prompt' and 'responses', and filters out entries with fewer than two responses.

    :param n_samples: Optional; the number of samples to select from the dataset. If None, all samples are used.
    :param seed: The seed for shuffling the dataset. Default is 0.
    :return: A HuggingFace Dataset object with processed entries from the UltraFeedback dataset.
    """
    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    if n_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))
    dataset = dataset.map(
        lambda x: {"prompt": x["instruction"], "responses": [c["response"] for c in x["completions"]]},
        num_proc=8
    )
    dataset = dataset.filter(
        lambda x: len(x["responses"]) >= 2,
        num_proc=8
    )
    return dataset


class SamplingStrategy(Enum):
    random = "random"
    keep_start = "keep_start"
    keep_end = "keep_end"


@dataclass
class MixDatasetConfig:
    ratio: float = 1.0
    sampling_strategy: SamplingStrategy = SamplingStrategy.random


@dataset_store(name="MixDataset")
def mix_datasets(
        datasets: Dict[str, Dataset],
        mix_configs: Dict[str, MixDatasetConfig]
):
    """
    Similar to load_concat_datasets, but add additional `mix_configs` to enable sampling before concat.

    :param datasets: A dict of HuggingFace dataset.
    :param mix_configs: A dict of MixDatasetConfig, the key should match exactly with `datasets` param.
    :return:
    """
    sampled_dataset_list = []

    assert len(datasets) == len(mix_configs)
    assert all(k in mix_configs.keys() for k in datasets.keys())

    for ds_name in datasets.keys():
        dataset = datasets[ds_name]
        config = mix_configs[ds_name]

        # perform sampling according to the config
        assert config.ratio <= 1.0
        n_example = max(math.floor(config.ratio * len(dataset)), 1)
        if config.ratio == 1.0:
            sampled_dataset = dataset
        elif config.sampling_strategy == SamplingStrategy.random:
            sampled_dataset = dataset.select(
                random.Random(0).sample(range(len(dataset)), k=n_example)
            )
        elif config.sampling_strategy == SamplingStrategy.keep_start:
            sampled_dataset = dataset.select(range(n_example))
        elif config.sampling_strategy == SamplingStrategy.keep_end:
            sampled_dataset = dataset.select(range(len(dataset) - n_example, len(dataset)))
        else:
            raise NotImplementedError

        sampled_dataset_list.append(sampled_dataset)

    return concatenate_datasets(sampled_dataset_list)
