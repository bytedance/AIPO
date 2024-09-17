# -*- coding: utf-8 -*-
# @Time    : 2024/2/6
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : chat_dataset.py

import itertools
import json
from typing import Union, List

import pyarrow
from datasets import Dataset
from mm_video.config import dataset_store

"""
Some chat datasets.
"""


@dataset_store()
class FastChat2OpenAssistantChatDataset(Dataset):
    """
    Custom Dataset subclass for converting chat data from FastChat format to OpenAssistant format supported by
    trl.SFTTrainer.
    """

    @staticmethod
    def load_from_file(file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def __init__(self, train_data: Union[str, List[str]]):
        """
        Initializes the dataset by loading the data from a JSON file and converting it to the appropriate format.

        :param train_data: The path to the JSON file or a list of JSON strings containing chat data.
        """
        if isinstance(train_data, list):
            data = itertools.chain.from_iterable(self.load_from_file(file_path) for file_path in train_data)
        else:
            data = self.load_from_file(train_data)

        # Convert to oai format supported by trl.SFTTrainer
        self.train_data = [
            {
                "messages": [
                    {
                        "role": conv["from"],
                        "content": conv["value"]
                    } for conv in json_obj["conversations"]
                ]
            } for json_obj in data
        ]
        # noinspection PyArgumentList
        super().__init__(arrow_table=pyarrow.Table.from_pylist(self.train_data))
