# -*- coding: utf-8 -*-
# @Time    : 8/7/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : plot_evaluation.py

import argparse
from pathlib import Path
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from alignment.tools.evaluation.summarize_evaluation import summarize_evaluation


def lim(s: str) -> Tuple[int, int]:
    nums = [int(num) for num in s.split(",") if num.isdigit()]
    assert len(nums) == 2, "Please provide two integers separated by a comma."
    if not nums:
        raise ValueError("No valid integer found in the input string")
    return min(nums), max(nums)


def read_all_tables(log_directories, pretty_names, max_iter=None):
    tables = {}
    for log_dir, pretty_name in zip(log_directories, pretty_names):
        table = summarize_evaluation(log_dir)
        if max_iter is not None:
            table['IterationNum'] = table['Iteration'].str.extract(r'(\d+)').astype(int)
            table = table[table['IterationNum'] < max_iter]
            # Drop the helper column
            table = table.drop(columns=['IterationNum'])
        tables[pretty_name] = table
    return tables


def plot_performance(log_directories, pretty_names, column_names, output_filepath='output.png', max_iter=None):
    # Set a beautiful style for the plots
    sns.set_theme(style="darkgrid")  # You can change to other styles like "darkgrid", "white", "ticks", etc.

    # Read all tables from log directories
    tables = read_all_tables(log_directories, pretty_names, max_iter=max_iter)

    # Get the list of all columns (metrics) from the column_names dictionary
    all_metrics = list(column_names.keys())

    # Set up the figure for subplots, with 2x3 size for each subplot
    num_metrics = len(all_metrics)
    col = num_metrics // 2 + num_metrics % 2
    fig, axes = plt.subplots(2, col, figsize=(4 * col, 2.5 * 2), dpi=300)

    # Define a palette for distinctive colors
    palette = sns.color_palette("husl", n_colors=len(pretty_names))

    first_legend_handles = None

    for metric_index, metric in enumerate(all_metrics):
        ax = axes.flatten()[metric_index] if num_metrics > 1 else axes
        for i, (pretty_name, table) in enumerate(tables.items()):
            # Ensure 'Iteration' and metric column exist in the table
            if 'Iteration' not in table.columns or metric not in table.columns:
                print(f"Metric {metric} not found in the data from {pretty_name}.")
                continue

            plot_df = table[['Iteration', metric]].dropna()
            plot_df["Iteration"] = plot_df["Iteration"].str.extract(r'(\d+)')[0].astype(int)
            plot_df = plot_df.sort_values('Iteration')

            # Error bars for specific metrics
            if metric == 'AlpacaEval (win %)' and 'AlpacaEval (std)' in table.columns:
                plot_df['std'] = table['AlpacaEval (std)']
                yerr = plot_df['std'] * 1.96  # 95% confidence interval
            elif metric == 'AlpacaEval 2.0 (win %)' and 'AlpacaEval 2.0 (std)' in table.columns:
                plot_df['std'] = table['AlpacaEval 2.0 (std)']
                yerr = plot_df['std'] * 1.96  # 95% confidence interval
            elif metric == 'Arena-Hard (win %)' and 'Arena-Hard (CI lower)' in table.columns and 'Arena-Hard (CI upper)' in table.columns:
                plot_df['ci_lower'] = table['Arena-Hard (CI lower)'].astype(float)
                plot_df['ci_upper'] = table['Arena-Hard (CI upper)'].astype(float)
                yerr = [plot_df['ci_lower'].abs(), plot_df['ci_upper'].abs()]  # Calculate yerr correctly
            else:
                yerr = None  # No error bars for this metric

            # Plot the line and error bars
            sns.lineplot(data=plot_df, x='Iteration', y=metric, label=pretty_name, ax=ax,
                         color=palette[i % len(palette)], linewidth=1.5, marker='o', markersize=5,
                         markeredgewidth=0.5)
            if yerr is not None:
                ax.errorbar(plot_df['Iteration'], plot_df[metric], yerr=yerr, fmt='none',
                            ecolor=palette[i % len(palette)], elinewidth=0.75, capsize=2, alpha=1.0)

            ax.set_xlabel("")
            ax.set_ylabel(column_names.get(metric, metric))

            # Collect legend handles from the first subplot
            if metric_index == 0:
                first_legend_handles, _ = ax.get_legend_handles_labels()
            if ax.legend_ is not None:
                ax.legend_.remove()  # Remove legend from the current subplot

            # Ensure the x-axis only shows Integers
            ax.set_xticks(plot_df["Iteration"].unique())
            ax.set_xticklabels([str(x + 1) for x in plot_df["Iteration"].unique().astype(int)])

            ax.yaxis.set_tick_params(labelsize=10, pad=0)
            ax.xaxis.set_tick_params(labelsize=9, pad=0)

    fig.align_ylabels(axes.flatten())

    # Add a single legend at the bottom of the fig
    fig.legend(handles=first_legend_handles, loc='lower center', ncol=len(pretty_names), columnspacing=1.0)

    plt.tight_layout(rect=[0, 0.06, 1, 1], w_pad=0.3)  # Adjust layout to add space for legend at the bottom

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0.01)


def plot_by_length(
        log_directories,
        pretty_names,
        output_filepath,
        plot_arena_hard_only=False,
        fit_lines=False,
        order=1,
        logx: bool = False,
        robust: bool = False,
        max_iter: Optional[int] = None,
        arena_hard_xlim: Optional[Tuple[int, int]] = None,
        arena_hard_ylim: Optional[Tuple[int, int]] = None,
        alpaca_eval_xlim: Optional[Tuple[int, int]] = None,
        alpaca_eval_ylim: Optional[Tuple[int, int]] = None,
        legend_on: str = "alpaca_eval"
):
    sns.set_theme(style="darkgrid")

    tables = read_all_tables(log_directories, pretty_names, max_iter)

    # Dynamically decide fig height based on row of legend
    legend_height_per_entry = 0.13
    num_legends = len(log_directories)
    col = 1
    row = num_legends // col + (num_legends % col != 0)
    additional_height = 0.15 + (row * legend_height_per_entry)

    if plot_arena_hard_only:
        subplot_height = 4
        fig, ax1 = plt.subplots(1, 1, figsize=(5, subplot_height + additional_height), dpi=300)
    else:
        subplot_height = 6
        fig, axes = plt.subplots(2, 1, figsize=(5, subplot_height + additional_height), dpi=300)
        ax1, ax2 = axes

    palette = sns.color_palette("husl", n_colors=len(pretty_names))

    all_handles = {}

    if logx:
        regplot_kwargs = dict(logx=True)
    elif robust:
        regplot_kwargs = dict(robust=True)
    else:
        regplot_kwargs = dict(order=order)

    # Plot Arena-Hard win rate vs. token length
    for i, (pretty_name, table) in enumerate(tables.items()):
        if 'Arena-Hard (avg_token)' not in table.columns or 'Arena-Hard (win %)' not in table.columns:
            print(f"Required columns not found in the data from {pretty_name}.")
            continue

        plot_df = table[
            ['Arena-Hard (avg_token)', 'Arena-Hard (win %)', 'Arena-Hard (CI lower)', 'Arena-Hard (CI upper)']].dropna()
        plot_df = plot_df.sort_values('Arena-Hard (avg_token)')

        if fit_lines:
            sns.regplot(
                data=plot_df,
                x='Arena-Hard (avg_token)', y='Arena-Hard (win %)',
                scatter=True, label=pretty_name, color=palette[i % len(palette)],
                ax=ax1,
                marker="x", scatter_kws={"s": 20}, line_kws={"lw": 2},
                # **regplot_kwargs
            )
        else:
            sns.lineplot(
                data=plot_df,
                x='Arena-Hard (avg_token)', y='Arena-Hard (win %)',
                label=pretty_name, ax=ax1, color=palette[i % len(palette)], linewidth=2,
                marker='o', markersize=5, markeredgewidth=0
            )

        handles, labels = ax1.get_legend_handles_labels()
        all_handles[pretty_name] = handles[-1]
        if ax1.legend_ is not None:
            ax1.legend_.remove()

        ax1.yaxis.set_major_locator(MultipleLocator(5))

        if arena_hard_xlim is not None:
            ax1.set_xlim(*arena_hard_xlim)
        if arena_hard_ylim is not None:
            ax1.set_ylim(*arena_hard_ylim)

    ax1.set_xlabel('Arena-Hard Avg Tokens', fontdict={'size': 13})
    ax1.set_ylabel('Arena-Hard WR (%)', fontdict={'size': 13})

    if not plot_arena_hard_only:
        for i, (pretty_name, table) in enumerate(tables.items()):
            if 'AlpacaEval 2.0 LC (win %)' not in table.columns or 'AlpacaEval 2.0 (avg_len)' not in table.columns:
                print(f"Required columns not found in the data from {pretty_name}.")
                continue

            plot_df = table[['AlpacaEval 2.0 (avg_len)', 'AlpacaEval 2.0 LC (win %)', 'AlpacaEval 2.0 (std)']].dropna()
            plot_df = plot_df.sort_values('AlpacaEval 2.0 (avg_len)')

            if fit_lines:
                sns.regplot(
                    data=plot_df,
                    x='AlpacaEval 2.0 (avg_len)', y='AlpacaEval 2.0 LC (win %)',
                    scatter=True, label=pretty_name, color=palette[i % len(palette)],
                    ax=ax2,
                    marker="x", scatter_kws={"s": 30}, line_kws={"lw": 2},
                    **regplot_kwargs
                )
            else:
                sns.lineplot(
                    data=plot_df,
                    x='AlpacaEval 2.0 (avg_len)', y='AlpacaEval 2.0 LC (win %)',
                    label=pretty_name, ax=ax2, color=palette[i % len(palette)], linewidth=2,
                    marker='o', markersize=5, markeredgewidth=0
                )

            handles, labels = ax2.get_legend_handles_labels()
            all_handles[pretty_name] = handles[-1]
            if ax2.legend_ is not None:
                ax2.legend_.remove()

            if alpaca_eval_xlim is not None:
                ax2.set_xlim(*alpaca_eval_xlim)
            if alpaca_eval_ylim is not None:
                ax2.set_ylim(*alpaca_eval_ylim)

        ax2.set_xlabel('AlpacaEval Avg Length', fontdict={'size': 13})
        ax2.set_ylabel('AlpacaEval 2.0 LC (%)', fontdict={'size': 13})

        fig.align_ylabels([ax1, ax2])

    # Add legend at one of the subplot
    if legend_on == "alpaca_eval":
        ax_with_legend = ax2
    elif legend_on == "arena_hard":
        ax_with_legend = ax1
    else:
        raise ValueError(f"Invalid legend_on value: {legend_on}")
    ax_with_legend.legend(
        handles=all_handles.values(), loc='lower right', ncol=col,
        handletextpad=0.1, labelspacing=0.1, handlelength=0.8, handleheight=1.2,
        columnspacing=1.0, fontsize=9
    )

    bottom_padding = additional_height / (subplot_height + additional_height)
    plt.tight_layout(rect=(0, bottom_padding, 1, 1), w_pad=0.01, h_pad=0.2)

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0.01)


def main():
    parser = argparse.ArgumentParser(description="Plot performance across iterations for multiple training logs.")
    parser.add_argument("--log_directories", nargs="+", help="List of log directory paths.", required=True)
    parser.add_argument("--pretty_names", nargs="+", help="List of pretty names corresponding to the log directories.")
    parser.add_argument("--output", default="output.png", help="Output image file path, default is 'output.png'.")
    parser.add_argument("--columns", nargs="*", help="Columns to plot (default: all columns will be plotted).")
    parser.add_argument("--pretty_columns", nargs="*", help="Pretty names for columns to be displayed on the plots.")
    parser.add_argument("--max-iter", type=int, help="Maximum number of iterations to plot.", default=None)
    parser.add_argument(
        "--by-length",
        action="store_true",
        help="Plot Arena-Hard win rate by token length."
    )
    parser.add_argument(
        "--fit-lines",
        action="store_true",
        help="Only valid when --by-length is specified. Fit regression lines."
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Only valid when both --by-length and --fit-lines are specified. Order of the regression line."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Only valid when both --by-length and --fit-lines are specified. Use log-linear regression."
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Only valid when both --by-length and --fit-lines are specified. Use robust regression."
    )
    parser.add_argument(
        "--only-arena-hard",
        action="store_true",
        help="Only valid when --by-length is specified. Plot only Arena-Hard data."
    )
    parser.add_argument(
        "--arena-hard-xlim",
        type=lim,
        help="Only valid when --by-length is specified. Set the x-axis limits for Arena-Hard plot."
    )
    parser.add_argument(
        "--arena-hard-ylim",
        type=lim,
        help="Only valid when --by-length is specified. Set the y-axis limits for Arena-Hard plot."
    )
    parser.add_argument(
        "--alpaca-eval-xlim",
        type=lim,
        help="Only valid when --by-length is specified. Set the x-axis limits for AlpacaEval 2.0 plot."
    )
    parser.add_argument(
        "--alpaca-eval-ylim",
        type=lim,
        help="Only valid when --by-length is specified. Set the y-axis limits for AlpacaEval 2.0 plot."
    )
    parser.add_argument(
        "--legend-on",
        choices=["alpaca_eval", "arena_hard"],
        default="alpaca_eval",
        help="Only valid when --by-length is specified. Plot legend on the specified plot."
    )

    args = parser.parse_args()

    if args.pretty_names:
        assert len(args.log_directories) == len(
            args.pretty_names), "The number of pretty names must match the number of log directories."
    else:
        args.pretty_names = ["Log {}".format(i + 1) for i in range(len(args.log_directories))]

    if args.columns and args.pretty_columns:
        assert len(args.columns) == len(
            args.pretty_columns), "The number of pretty columns must match the number of columns."
        column_names = dict(zip(args.columns, args.pretty_columns))
    else:
        # Default columns and pretty names
        column_names = {
            # 'MT-Bench (GPT-4)': 'MT-Bench (GPT-4)',
            'MT-Bench (GPT-4-Turbo)': 'MT-Bench (GPT-4-Turbo)',
            'Arena-Hard (win %)': 'Arena-Hard WR (%)',
            'Arena-Hard (avg_token)': 'Arena-Hard Avg Tokens',
            # 'AlpacaEval (win %)': 'AlpacaEval (win %)',
            'AlpacaEval 2.0 LC (win %)': 'AlpacaEval 2.0 LC (%)',
            'AlpacaEval 2.0 (win %)': 'AlpacaEval 2.0 WR (%)',
            'AlpacaEval 2.0 (avg_len)': 'AlpacaEval Avg Length',
        }

    if args.by_length:
        plot_by_length(
            log_directories=args.log_directories,
            pretty_names=args.pretty_names,
            output_filepath=args.output,
            plot_arena_hard_only=args.only_arena_hard,
            fit_lines=args.fit_lines,
            order=args.order,
            logx=args.log,
            robust=args.robust,
            max_iter=args.max_iter,
            arena_hard_xlim=args.arena_hard_xlim, arena_hard_ylim=args.arena_hard_ylim,
            alpaca_eval_xlim=args.alpaca_eval_xlim, alpaca_eval_ylim=args.alpaca_eval_ylim,
            legend_on=args.legend_on
        )
    else:
        plot_performance(args.log_directories, args.pretty_names, column_names, args.output, args.max_iter)


if __name__ == "__main__":
    main()
