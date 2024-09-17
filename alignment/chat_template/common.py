# -*- coding: utf-8 -*-
# @Time    : 5/30/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : common.py

__all__ = [
    "get_chat_template",
    "use_chat_template"
]

import logging
import os
from typing import Optional

from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def get_chat_template(chat_template: str) -> str:
    """
    Get the chat template from the file.

    :param chat_template: Name of the chat template to use.
    :return: The chat template as a string.
    """
    model_template = os.path.join(os.path.dirname(__file__), f"{chat_template}.jinja")
    with open(model_template, "r") as f:
        return f.read().replace('    ', '').replace('\n', '')


def use_chat_template(chat_template: Optional[str], tokenizer: PreTrainedTokenizer) -> None:
    """
    Set the chat template for the tokenizer. Will update the tokenizer **in-place**.

    :param chat_template: Name of the chat template to use.
    :param tokenizer: The tokenizer to set the chat template for.
    :return: None
    """
    if chat_template is not None:
        tokenizer.chat_template = get_chat_template(chat_template)
    elif tokenizer.chat_template is None:
        logger.warning(
            "No chat template provided and no existing template found in the tokenizer. The tokenizer may not "
            "function as expected without a chat template."
        )
