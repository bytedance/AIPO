# -*- coding: utf-8 -*-
# @Time    : 8/1/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : generate_arena_hard_output.py

"""Generate answers using api endpoints.

Usage:
python gen_api_answer --parallel 32
"""
import argparse
import json
import os
import time
from typing import List

import shortuuid
import tiktoken
import torch.cuda
from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    temperature_config
)
from mm_video.utils.language.generate import VLLMGenerator
from tqdm import tqdm
from vllm import SamplingParams

from alignment.chat_template import get_chat_template
from alignment.tools.evaluation.generate_arena_hard_judgment import make_config
from alignment.tools.evaluation.generate_mt_bench_output import reorg_answer_file


def get_answer(
        question: dict,
        model_id: str,
        generator: VLLMGenerator,
        tokenizer,
        num_choices: int,
        max_tokens: int,
        temperature: float,
        answer_file: str
):
    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]

    conv = []

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    choices = []
    for i in range(num_choices):
        turns = []
        for j in range(len(question["turns"])):
            conv.append({"role": "user", "content": question["turns"][j]["content"]})
            prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            output = generator.generate([prompt], max_tokens=max_tokens, temperature=temperature)[0]
            output = output.strip()

            conv.append({"role": "assistant", "content": output})

            turns.append({"content": output, "token_len": len(encoding.encode(output, disallowed_special=()))})
        choices.append({"index": i, "turns": turns})

    # Dump answers
    ans = {
        "question_id": question["question_id"],
        "answer_id": shortuuid.uuid(),
        "model_id": model_id,
        "choices": choices,
        "tstamp": time.time(),
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a") as fout:
        fout.write(json.dumps(ans) + "\n")


def get_answer_batch(
        questions: List[dict],
        model_id: str,
        generator: VLLMGenerator,
        tokenizer,
        num_choices: int,
        max_tokens_list: List[int],
        temperature: float,
        answer_file: str
):
    assert all(len(q["turns"]) == 1 for q in questions)
    temperatures = [
        temperature_config[q["category"]] if q["category"] in temperature_config else temperature
        for q in questions
    ]

    prompts = []
    sampling_params = []
    for question, temperature, max_tokens in zip(questions, temperatures, max_tokens_list):
        conv = []
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        for i in range(num_choices):
            conv.append({"role": "user", "content": question["turns"][0]["content"]})
            prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            sampling_params.append(SamplingParams(max_tokens=max_tokens, temperature=temperature))

    print("Prompt:")
    print(prompts[0])
    print("<<< End of Prompt <<<<<")
    outputs = generator.generate(prompts, generate_kwargs=dict(sampling_params=sampling_params))

    count = 0
    for question in questions:
        choices = []
        for i in range(num_choices):
            turns = []

            output = outputs[count]
            count += 1
            output = output.strip()

            conv.append({"role": "assistant", "content": output})

            turns.append({"content": output, "token_len": len(encoding.encode(output, disallowed_special=()))})
            choices.append({"index": i, "turns": turns})

        # Dump answers
        ans = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": choices,
            "tstamp": time.time(),
        }

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")


def none_or_str(value: str):
    if value == 'None' or value == "null":
        return None
    return value


def none_or_int(value: str):
    if value == 'None' or value == "null":
        return None
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--chat-template",
        type=none_or_str,
        default=None,
        help="Specifying chat template for generation."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="The path to the root directory for saving evaluation outputs."
    )
    parser.add_argument(
        "--batching",
        action="store_true",
        help="Batching generation."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Dtype for vllm generation."
    )
    parser.add_argument(
        "--setting-file", type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "arena-hard-v0.1", "gen_answer_config.yaml")
    )
    parser.add_argument(
        "--max-model-len",
        type=none_or_int,
        default=None,
        help="max_model_len for vllm generation."
    )
    args = parser.parse_args()

    args.model_id = args.model_id.replace("/", "_")

    settings = make_config(args.setting_file)

    existing_answer = load_model_answers(os.path.join(args.data_path, "arena-hard-v0.1", "model_answer"))

    print(settings)

    question_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "arena-hard-v0.1", "question.jsonl")
    questions = load_questions(question_file, None, None)

    answer_file = os.path.join(args.data_path, "arena-hard-v0.1", "model_answer", f"{args.model_id}.jsonl")
    print(f"Output to {answer_file}")

    # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
    question_list = [question["turns"][0]["content"] for question in questions]
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.chat_template is not None:
        tokenizer.chat_template = get_chat_template(chat_template=args.chat_template)

    tokens = tokenizer(question_list)
    max_tokens = [(settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]

    count = 0
    generator = VLLMGenerator(
        args.model_path,
        generate_kwargs=dict(use_tqdm=args.batching),
        model_init_kwargs=dict(
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=args.dtype,
            max_model_len=args.max_model_len
        )
    )

    if not args.batching:
        for index, question in enumerate(tqdm(questions, desc="Generating")):
            if args.model_id in existing_answer and question["question_id"] in existing_answer[args.model_id]:
                count += 1
                continue
            get_answer(
                question=question,
                model_id=args.model_id,
                generator=generator,
                tokenizer=tokenizer,
                num_choices=settings["num_choices"],
                max_tokens=max_tokens[index],
                temperature=settings["temperature"],
                answer_file=answer_file,
            )
    else:
        questions_ = []
        max_tokens_list = []

        for index, question in enumerate(questions):
            if args.model_id in existing_answer and question["question_id"] in existing_answer[args.model_id]:
                count += 1
                continue
            questions_.append(question)
            max_tokens_list.append(max_tokens[index])

        if questions_:
            get_answer_batch(
                questions=questions_,
                model_id=args.model_id,
                generator=generator,
                tokenizer=tokenizer,
                num_choices=settings["num_choices"],
                max_tokens_list=max_tokens_list,
                temperature=settings["temperature"],
                answer_file=answer_file,
            )

    if count > 0:
        print(f"{count} number of existing answers")

    reorg_answer_file(answer_file)


if __name__ == "__main__":
    main()
