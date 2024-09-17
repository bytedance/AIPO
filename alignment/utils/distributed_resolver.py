# -*- coding: utf-8 -*-
# @Time    : 2024/3/29
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : distributed_resolver.py

"""
This is a resolver for OmegaConf.
It supports obtaining the parameters required for running torch distributed by
parsing environment variables and can be used in conjunction with the
hydra-torchrun-launcher (https://github.com/acherstyx/hydra-torchrun-launcher).
"""

import os
from ipaddress import ip_address, IPv4Address

from omegaconf import OmegaConf


def is_ipv6(addr: str) -> bool:
    return not (type(ip_address(addr)) is IPv4Address)


def dist_nproc_per_node() -> int:
    return int(os.environ["DIST_NPROC_PER_NODE"])


def dist_nnodes() -> int:
    return int(os.environ["DIST_NNODES"])


def dist_node_rank() -> int:
    return int(os.environ["DIST_NODE_RANK"])


def dist_master_addr() -> str:
    addr = os.environ["DIST_MASTER_ADDR"]
    if is_ipv6(addr):
        return f"[{addr}]"
    else:
        return addr


def dist_master_port() -> int:
    """
    Support multiple port seperated by comma, will use the first port.
    :return:
    """
    return int(os.environ["DIST_MASTER_PORT"].split(",")[0])


OmegaConf.register_new_resolver("dist.nproc_per_node", dist_nproc_per_node)
OmegaConf.register_new_resolver("dist.nnodes", dist_nnodes)
OmegaConf.register_new_resolver("dist.node_rank", dist_node_rank)
OmegaConf.register_new_resolver("dist.master_addr", dist_master_addr)
OmegaConf.register_new_resolver("dist.master_port", dist_master_port)
