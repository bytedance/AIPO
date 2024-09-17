# -*- coding: utf-8 -*-
# @Time    : 9/10/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : hf_trainer_state.py

from typing import List, Dict, Tuple

from mm_video.utils.common import load_json


def load_log_history(log_root: str) -> List[Dict[str, float]]:
    """
    :param log_root: dpo trainer log root
    :return:
    """
    trainer_state = load_json(f'{log_root}/trainer_state.json')
    return trainer_state["log_history"]


def get_metric(log_history: List[Dict[str, float]], metric_name: str) -> Tuple[List[int], List[float]]:
    """
    Get metric from trainer log history. Will return a list of step and corresponding metric value.

    :param log_history: log history, see `load_log_history`
    :param metric_name: name of the metric
    :return:
    """
    steps: List[int] = []
    metrics: List[float] = []
    for item in log_history:
        if metric_name in item:
            steps.append(int(item["step"]))
            metrics.append(item[metric_name])
    return steps, metrics


def smooth_data(data: List[float], alpha: float = 0.3) -> List[float]:
    smoothed = []
    for i, point in enumerate(data):
        if i == 0:
            smoothed.append(point)
        else:
            smoothed.append(alpha * point + (1 - alpha) * smoothed[-1])
    return smoothed
