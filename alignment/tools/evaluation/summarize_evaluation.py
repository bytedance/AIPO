# -*- coding: utf-8 -*-
# @Time    : 7/29/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : summarize_evaluation.py

import argparse
import os

import pandas as pd


def parse_alpaca_eval(filepath, version="1.0"):
    if not os.path.exists(filepath):
        if version == "1.0":
            return pd.DataFrame(columns=['Iteration', 'AlpacaEval (win %)', 'AlpacaEval (std)', 'AlpacaEval (n_total)',
                                         'AlpacaEval 1.0 (avg_len)'])
        else:
            return pd.DataFrame(
                columns=['Iteration', 'AlpacaEval 2.0 LC (win %)', 'AlpacaEval 2.0 (win %)', 'AlpacaEval 2.0 (std)',
                         'AlpacaEval 2.0 (n_total)', 'AlpacaEval 2.0 (avg_len)'])

    data = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) == 0 or "win_rate" in parts:
                continue  # Skip header or empty lines
            if version == "1.0":
                if len(parts) == 5:
                    iteration = parts[0].split('/')[-1]
                    win_rate = float(parts[1])
                    std_err = float(parts[2])
                    n_total = int(parts[3])
                    avg_length = int(parts[4])
                    data[iteration] = (win_rate, std_err, n_total, avg_length)
            elif version == "2.0":
                if len(parts) == 6:
                    iteration = parts[0].split('/')[-1]
                    lc_win_rate = float(parts[1])
                    win_rate = float(parts[2])
                    std_err = float(parts[3])
                    n_total = int(parts[4])
                    avg_length = int(parts[5])
                    data[iteration] = (lc_win_rate, win_rate, std_err, n_total, avg_length)

    if version == "1.0":
        df = pd.DataFrame(
            [(k, v[0], v[1], v[2], v[3]) for k, v in data.items()],
            columns=['Iteration', 'AlpacaEval (win %)', 'AlpacaEval (std)', 'AlpacaEval (n_total)',
                     'AlpacaEval 1.0 (avg_len)']
        )
    else:
        df = pd.DataFrame(
            [(k, v[0], v[1], v[2], v[3], v[4]) for k, v in data.items()],
            columns=['Iteration', 'AlpacaEval 2.0 LC (win %)', 'AlpacaEval 2.0 (win %)', 'AlpacaEval 2.0 (std)',
                     'AlpacaEval 2.0 (n_total)', 'AlpacaEval 2.0 (avg_len)']
        )

    return df


def parse_mt_bench(filepath, model_name="MT-Bench (GPT-4)"):
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['Iteration', model_name])

    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Find the last occurrence of "########## Average ##########"
    last_avg_idx = None
    for i in range(len(lines)):
        if "########## Average ##########" in lines[i]:
            last_avg_idx = i

    # If the marker is found, start collecting data from there
    if last_avg_idx is not None:
        for line in lines[last_avg_idx + 1:]:
            if line.startswith("##########"):
                break
            if "score" not in line:
                parts = line.split()
                if len(parts) == 2:
                    model = parts[0].split('_')[-1]
                    try:
                        score = float(parts[1])
                        data.append((model, score))
                    except ValueError:
                        continue

    return pd.DataFrame(data, columns=['Iteration', model_name])


def parse_arena_hard(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame(
            columns=['Iteration', 'Arena-Hard (win %)', 'Arena-Hard (CI lower)', 'Arena-Hard (CI upper)',
                     'Arena-Hard (avg_token)'])

    data = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if "score:" in line:
            parts = line.split('|')
            if len(parts) == 4:
                model_info = parts[0].strip().split()[-1]
                if 'iter' not in model_info:
                    continue
                iteration = model_info.split('_')[-1]
                try:
                    score = float(parts[1].split(':')[1].strip())
                    ci_values = parts[2].split(':')[1].strip().strip('()').split(',')
                    ci_lower = float(ci_values[0].strip())
                    ci_upper = float(ci_values[1].strip())
                    avg_tokens = int(parts[3].split(':')[1].strip())
                    data[iteration] = (score, ci_lower, ci_upper, avg_tokens)
                except ValueError:
                    continue

    df = pd.DataFrame(
        [(k, v[0], v[1], v[2], v[3]) for k, v in data.items()],
        columns=['Iteration', 'Arena-Hard (win %)', 'Arena-Hard (CI lower)', 'Arena-Hard (CI upper)',
                 'Arena-Hard (avg_token)']
    )

    return df


def generate_summarize_table(mt_bench, mt_bench_gpt_4_turbo, alpaca_eval, alpaca_eval_2, arena_hard):
    combined_df = pd.merge(mt_bench, mt_bench_gpt_4_turbo, on='Iteration', how='outer')
    combined_df = pd.merge(combined_df, alpaca_eval, on='Iteration', how='outer')
    combined_df = pd.merge(combined_df, alpaca_eval_2, on='Iteration', how='outer')
    combined_df = pd.merge(combined_df, arena_hard, on='Iteration', how='outer')
    combined_df = combined_df.sort_values(by='Iteration')
    return combined_df


def summarize_evaluation(log_directory):
    alpaca_eval_file = os.path.join(log_directory, "evaluation", "alpaca_eval.txt")
    alpaca_eval_2_file = os.path.join(log_directory, "evaluation", "alpaca_eval_2.txt")
    mt_bench_file = os.path.join(log_directory, "evaluation", "mt_bench.txt")
    mt_bench_gpt_4_turbo_file = os.path.join(log_directory, "evaluation", "mt_bench_gpt_4_turbo.txt")
    arena_hard_file = os.path.join(log_directory, "evaluation", "arena_hard.txt")

    alpaca_eval = parse_alpaca_eval(alpaca_eval_file, version="1.0")
    alpaca_eval_2 = parse_alpaca_eval(alpaca_eval_2_file, version="2.0")
    mt_bench = parse_mt_bench(mt_bench_file, "MT-Bench (GPT-4)")
    mt_bench_gpt_4_turbo = parse_mt_bench(mt_bench_gpt_4_turbo_file, "MT-Bench (GPT-4-Turbo)")
    arena_hard = parse_arena_hard(arena_hard_file)

    summarize_table = generate_summarize_table(mt_bench, mt_bench_gpt_4_turbo, alpaca_eval, alpaca_eval_2, arena_hard)

    # order by iteration
    summarize_table['Iteration'] = summarize_table['Iteration'].str.replace('iter', '').astype(int)
    summarize_table = summarize_table.sort_values(by='Iteration').reset_index(drop=True)
    summarize_table['Iteration'] = 'iter' + summarize_table['Iteration'].astype(str)

    return summarize_table


def cli():
    parser = argparse.ArgumentParser(description="Summarize evaluation results from multiple log directories.")
    parser.add_argument("log_directories", nargs="+", help="List of log directory paths.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity")

    args = parser.parse_args()

    for log_directory in args.log_directories:
        print(f"Summarizing results for log directory: {log_directory}")
        summarize_table = summarize_evaluation(log_directory)

        # Check avg length consistency and add a warning if they are different
        if 'AlpacaEval 1.0 (avg_len)' in summarize_table.columns and 'AlpacaEval 2.0 (avg_len)' in summarize_table.columns:
            different_avg_lengths = summarize_table['AlpacaEval 1.0 (avg_len)'] != summarize_table[
                'AlpacaEval 2.0 (avg_len)']
            if different_avg_lengths.any():
                print("Warning: Different average lengths found between versions 1.0 and 2.0 for some iterations.")
                print(summarize_table.loc[
                          different_avg_lengths, ['Iteration', 'AlpacaEval 1.0 (avg_len)', 'AlpacaEval 2.0 (avg_len)']])
            summarize_table['AlpacaEval (avg_len)'] = summarize_table['AlpacaEval 1.0 (avg_len)'].combine_first(
                summarize_table['AlpacaEval 2.0 (avg_len)'])
            summarize_table.drop(columns=['AlpacaEval 1.0 (avg_len)', 'AlpacaEval 2.0 (avg_len)'], inplace=True)
        else:
            summarize_table['AlpacaEval (avg_len)'] = summarize_table['AlpacaEval 1.0 (avg_len)'].combine_first(
                summarize_table['AlpacaEval 2.0 (avg_len)'])
            summarize_table.drop(columns=['AlpacaEval 1.0 (avg_len)', 'AlpacaEval 2.0 (avg_len)'], inplace=True)

        # Ensure all columns are present
        for column in ['AlpacaEval (avg_len)', 'MT-Bench (GPT-4)', 'MT-Bench (GPT-4-Turbo)', 'AlpacaEval (win %)',
                       'AlpacaEval (std)', 'AlpacaEval (n_total)',
                       'AlpacaEval 2.0 LC (win %)', 'AlpacaEval 2.0 (win %)', 'AlpacaEval 2.0 (std)',
                       'AlpacaEval 2.0 (n_total)',
                       'Arena-Hard (win %)', 'Arena-Hard (CI lower)', 'Arena-Hard (CI upper)',
                       'Arena-Hard (avg_token)']:
            if column not in summarize_table.columns:
                summarize_table[column] = ""

        if args.verbose == 0:
            # Default columns
            column_order = [
                'Iteration',
                'MT-Bench (GPT-4-Turbo)',
                'MT-Bench (GPT-4)',
                'AlpacaEval (win %)',
                'AlpacaEval 2.0 LC (win %)',
                'AlpacaEval 2.0 (win %)',
                'AlpacaEval (avg_len)',
                'Arena-Hard (win %)',
                'Arena-Hard (avg_token)'
            ]
        elif args.verbose == 1:
            # More output columns
            column_order = [
                'Iteration',
                'MT-Bench (GPT-4-Turbo)',
                'MT-Bench (GPT-4)',
                'AlpacaEval (win %)',
                'AlpacaEval (std)',
                'AlpacaEval 2.0 LC (win %)',
                'AlpacaEval 2.0 (win %)',
                'AlpacaEval 2.0 (std)',
                'AlpacaEval (avg_len)',
                'Arena-Hard (win %)',
                'Arena-Hard (CI lower)',
                'Arena-Hard (CI upper)',
                'Arena-Hard (avg_token)'
            ]
        else:
            # Full table
            column_order = [
                'Iteration',
                'MT-Bench (GPT-4-Turbo)',
                'MT-Bench (GPT-4)',
                'AlpacaEval (win %)',
                'AlpacaEval (std)',
                'AlpacaEval (n_total)',
                'AlpacaEval 2.0 LC (win %)',
                'AlpacaEval 2.0 (win %)',
                'AlpacaEval 2.0 (std)',
                'AlpacaEval 2.0 (n_total)',
                'AlpacaEval (avg_len)',
                'Arena-Hard (win %)',
                'Arena-Hard (CI lower)',
                'Arena-Hard (CI upper)',
                'Arena-Hard (avg_token)'
            ]

        # Ensure only the specified columns are included in the final output
        final_column_order = [col for col in column_order if col in summarize_table.columns]

        summarize_table = summarize_table[final_column_order]

        summarize_table.fillna("", inplace=True)
        numeric_columns = summarize_table.select_dtypes(include=['float64']).columns
        summarize_table[numeric_columns] = summarize_table[numeric_columns].round(2)

        print(summarize_table.to_markdown(index=False))
        print()  # Separating each summary with a line


if __name__ == "__main__":
    cli()
