# -*- coding: utf-8 -*-
# @Time    : 5/8/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : generate_alpaca_eval_output.py

from typing import Optional

import datasets
import fire
import torch.cuda
from mm_video.utils.common import save_json
from mm_video.utils.language import VLLMGenerator, HFPipelineGenerator
from tqdm import tqdm

from alignment.runner.stages.stage_generate_responses import apply_chat_template
from alignment.runner.utils.common import get_tokenizer


def create_messages(example):
    example["messages"] = [{"role": "user", "content": example["instruction"]}]
    return example


def main(
        model_name_or_path: str,
        output: str,
        chat_template: Optional[str] = None,
        add_system_message: bool = False,
        pretty_name: Optional[str] = None,
        batching: bool = False,
        batch_size: int = 1,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        add_think_step_by_step: bool = False,
        use_vllm: bool = False,
        tensor_parallel_size: Optional[int] = None,
        dtype: str = "auto",
        max_model_len: Optional[int] = None
):
    """

    :param model_name_or_path:
    :param output:
    :param chat_template: Load chat template defined in this repo. If not specified, the default template of the
        tokenizer will be used.
    :param add_system_message: Add an empty system message to the start of conversation.
    :param pretty_name: Name to display on the leaderboard
    :param batching: Only valid for vllm generator when `use_vllm=True`, disable batching, generate one-by-one
    :param batch_size: Only valid for huggingface generator when `use_vllm=False`

    :param max_new_tokens: Generate kwargs
    :param temperature: Generate kwargs
    :param top_p: Generate kwargs
    :param top_k: Generate kwargs

    :param add_think_step_by_step:
    :param use_vllm: vllm/huggingface generator, will use huggingface generator by default
    :param tensor_parallel_size: Only valid for vllm generator when `use_vllm=True`, tensor parallel size
    :param dtype: Only valid for vllm generator when `use_vllm=True`, dtype
    :param max_model_len: Only valid for vllm generator when `use_vllm=True`, max_model_len
    """
    if pretty_name is None:
        pretty_name = model_name_or_path

    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count()

    if chat_template == "None" or chat_template == "null":
        chat_template = None
    if max_model_len == "None" or max_model_len == "null":
        max_model_len = None

    # Test dataset
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]

    tokenizer = get_tokenizer(model_name_or_path, chat_template)

    # Create prompt
    preprocessed_eval_set = eval_set.map(
        create_messages,
        keep_in_memory=True,
        num_proc=8,
        desc="Create messages"
    )
    preprocessed_eval_set = preprocessed_eval_set.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "add_system_message": add_system_message},
        keep_in_memory=True,
        num_proc=8,
        desc="Apply chat template"
    )
    prompts = preprocessed_eval_set[:]["prompt_with_chat_template"]
    if add_think_step_by_step:
        prompts = list(prompts)
        prompts = [p + "Let's think step by step. " for p in prompts]

    # Print the first prompt as example for check
    print(f"Prompt example:\n{prompts[0]}")
    print("<<< END OF EXAMPLE")

    # Generate
    # Generation config match with:
    # https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/models_configs/zephyr-7b-beta/configs.yaml
    # https://github.com/tatsu-lab/alpaca_eval/blob/4ced94f63cec6de3ac9e69ee59634d43e1bcba5c/src/alpaca_eval/decoders/huggingface_local.py#L28
    if use_vllm:
        generator = VLLMGenerator(
            model_name_or_path,
            model_init_kwargs=dict(
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                max_model_len=max_model_len,
            ),
            sampling_params_kwargs=dict(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        )
        if batching:
            responses = generator.generate(prompts)
        else:
            responses = [generator.generate([p], use_tqdm=False)[0] for p in tqdm(prompts)]
    else:
        generator = HFPipelineGenerator(
            model_name_or_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            generate_kwargs=dict(
                do_sample=True if temperature > 0 else False,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            ),
            pipeline_kwargs=dict(
                device="cuda"
            )
        )
        responses = generator.generate(prompts)

    eval_set = eval_set.remove_columns(["output", "generator"])
    eval_set = eval_set.add_column("output", responses)
    eval_set = eval_set.add_column("generator", [pretty_name] * len(eval_set))

    save_json(eval_set.to_list(), output, save_pretty=True)


def main_cli():
    fire.Fire(main)


if __name__ == '__main__':
    main_cli()
