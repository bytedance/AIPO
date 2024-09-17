# -*- coding: utf-8 -*-
# @Time    : 6/25/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : generate_mt_bench_output.py


"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

Generate answers with local models.

Usage:
TBD
"""

import argparse
import json
import os
import time

import shortuuid
import torch
from fastchat.llm_judge.common import load_questions, temperature_config
from mm_video.utils.language import VLLMGenerator
from tqdm import tqdm
from transformers import AutoTokenizer

from alignment.runner.utils.common import get_chat_template


@torch.inference_mode()
def get_model_answers(
        generator,
        tokenizer,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
):
    print_prompt_once = True

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            turns = []
            messages = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                messages.append({"role": "user", "content": qs})
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                if print_prompt_once:
                    print(f"Prompt:\n{prompt}")
                    print("<<<<< End of Prompt <<<<<")
                    print_prompt_once = False

                # some models may error out when generating long outputs
                try:
                    output = generator.generate(
                        [prompt],
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                    )[0]
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                messages.append({"role": "assistant", "content": output})
                turns.append(output)
            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


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
        "--model-id",
        type=str,
        required=True,
        help="A custom name for the model."
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
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--answer-file",
        type=str,
        help="The output answer file."
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16", "auto"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default="auto",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--max-model-len",
        type=none_or_int,
        default=None,
        help="max_model_len for vllm generation."
    )

    args = parser.parse_args()

    args.model_id = args.model_id.replace("/", "_")

    question_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.bench_name, "question.jsonl")
    answer_file = os.path.join(args.data_path, args.bench_name, "model_answer", f"{args.model_id}.jsonl")
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    print(f"Output to {answer_file}")

    questions = load_questions(question_file, args.question_begin, args.question_end)

    # Build generator and tokenizer
    generator = VLLMGenerator(
        args.model_path,
        generate_kwargs=dict(use_tqdm=False),
        model_init_kwargs=dict(
            tensor_parallel_size=torch.cuda.device_count(),
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            revision=args.revision,
        ),
        sampling_params_kwargs=dict(top_k=50)  # Match with HuggingFace default
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.chat_template is not None:
        tokenizer.chat_template = get_chat_template(chat_template=args.chat_template)

    get_model_answers(
        generator=generator,
        tokenizer=tokenizer,
        model_id=args.model_id,
        questions=questions,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
    )

    reorg_answer_file(answer_file)


if __name__ == "__main__":
    main()
