# -*- coding: utf-8 -*-
# @Time    : 7/31/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : generate_arena_hard_judgment.py

"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/lm-sys/arena-hard-auto/blob/main/gen_judgment.py
"""

import argparse
import concurrent.futures
import json
import os
import re
import time

import openai
import tiktoken
import yaml
from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    get_model_list,
)
from openai import AzureOpenAI
from tqdm import tqdm

# API setting constants
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"


# load config args from config yaml files
def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs = yaml.load(f, Loader=yaml.SafeLoader)

    return config_kwargs


def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False


def chat_completion_openai_azure(model, messages, temperature, max_tokens, api_dict=None):
    if api_dict is not None:
        azure_endpoint = api_dict["api_base"]
        api_key = api_dict["api_key"]
    else:
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_key = os.environ["AZURE_OPENAI_KEY"]
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2023-07-01-preview",
        azure_endpoint=azure_endpoint
    )

    if "azure-" in model:
        model = model[6:]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = response.choices[0].message.content
            break
        except openai.OpenAIError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(type(e), e)
            break
        except KeyError:
            print(response)
            break

    return output


# get answer from model
def get_answer(model, conv, temperature, max_tokens, endpoint_dict=None):
    output = chat_completion_openai_azure(model, conv, temperature, max_tokens)
    return output


def judgment(**args):
    question = args["question"]
    answer = args["answer"]
    reference = args["reference"]
    baseline = args["baseline_answer"]
    configs = args["configs"]
    output_file = args["output_file"]
    model = configs["judge_model"]

    num_games = 2 if configs["pairwise"] else 1

    output = {
        "question_id": question["question_id"],
        "model": answer["model_id"],
        "judge": model,
        "games": []
    }

    encoding = tiktoken.encoding_for_model("gpt-4")

    for game in range(num_games):
        conv = [{"role": "system", "content": configs["system_prompt"]}]

        for template in configs["prompt_template"]:
            prompt_args = {}

            for i, turn in enumerate(question["turns"]):
                prompt_args[f"question_{i + 1}"] = turn["content"]
            base = 1

            if baseline:
                if game % 2 == 1:  # swap position
                    answer, baseline = baseline, answer

                for i, turn in enumerate(baseline["choices"][0]["turns"]):
                    prompt_args[f"answer_{i + 1}"] = turn["content"]
                    base += 1
            if answer:
                for i, turn in enumerate(answer["choices"][0]["turns"]):
                    prompt_args[f"answer_{i + base}"] = turn["content"]

            if reference:
                for j, ref_answer in enumerate(reference):
                    for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                        prompt_args[f"ref_answer_{i + j + 1}"] = turn["content"]

            user_prompt = template.format(**prompt_args)
            conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(configs['number_of_judgment_attempts']):
            new_judgment = get_answer(
                model,
                conv,
                configs["temperature"],
                configs["max_tokens"],
            )

            judgment += ("\n" + new_judgment)

            score, try_again = get_score(judgment, args["regex_pattern"])

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append(
                {"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        result = {
            "user_prompt": conv[1]["content"],
            "judgment": judgment,
            "score": score,
            "n_token": len(encoding.encode(judgment, disallowed_special=()))
        }
        output["games"].append(result)

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="The path to the root directory for saving evaluation outputs."
    )
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--setting-file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "arena-hard-v0.1", "judge_config.yaml")
    )
    args = parser.parse_args()
    print(args)

    configs = make_config(args.setting_file)

    print(
        f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
        + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}')

    if configs["regex_pattern"]:
        pattern = re.compile(configs["regex_pattern"])

    question_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), configs["bench_name"], "question.jsonl")
    answer_dir = os.path.join(args.data_path, configs["bench_name"], "model_answer")
    base_answer_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), configs["bench_name"], "model_answer")
    ref_answer_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), configs["bench_name"],
                                  "reference_answer")

    questions = load_questions(question_file, None, None)
    model_answers = load_model_answers(answer_dir)
    baseline_model_answers = load_model_answers(base_answer_dir)
    model_answers.update(baseline_model_answers)

    # if user choose a set of models, only judge those models
    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list
        models = [m.replace("/", "_") for m in models]

    ref_answers = None
    if configs["reference"]:
        ref_answers = load_model_answers(ref_answer_dir)
        ref_answers = [ref_answers[model] for model in configs["ref_model"]]

    output_files = {}
    output_dir = os.path.join(args.data_path, configs['bench_name'], "model_judgment", configs['judge_model'])
    for model in models:
        output_files[model] = os.path.join(
            output_dir,
            f"{model}.jsonl",
        )

    for output_file in output_files.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_judgments = load_model_answers(output_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for model in models:
            count = 0
            for question in questions:
                question_id = question["question_id"]

                kwargs = {}
                kwargs["question"] = question
                if model in model_answers and not question_id in model_answers[model]:
                    print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                    continue

                if model in existing_judgments and question_id in existing_judgments[model]:
                    count += 1
                    continue

                kwargs["answer"] = model_answers[model][question_id]
                if ref_answers:
                    kwargs["reference"] = [ref_answer[question_id] for ref_answer in ref_answers]
                    assert len(kwargs["reference"]) == len(configs["ref_model"])
                else:
                    kwargs["reference"] = None
                if configs["baseline"]:
                    kwargs["baseline_answer"] = model_answers[configs["baseline_model"]][question_id]
                else:
                    kwargs["baseline_answer"] = None
                kwargs["configs"] = configs
                kwargs["output_file"] = output_files[model]
                kwargs["regex_pattern"] = pattern
                future = executor.submit(judgment, **kwargs)
                futures.append(future)

            if count > 0:
                print(f"{count} number of existing judgments")

        for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()


if __name__ == "__main__":
    main()
