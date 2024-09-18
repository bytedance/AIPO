# -*- coding: utf-8 -*-
# @Time    : 6/10/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : customized_dpo_trainer.py

import logging
from dataclasses import dataclass
from typing import Dict, Union, Literal, List, Tuple, Optional

import torch
import torch.nn.functional as F
from hydra_zen import builds, just
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)

"""
This implementation is originally based on trl==0.9.3, may not compatible with other versions.
"""


@dataclass
class CustomizedDPOConfig(DPOConfig):
    # Hyperparameters
    alpha: Optional[float] = None
    alpha1: Optional[float] = None
    alpha2: Optional[float] = None
    lambda_: Optional[float] = None
    lambda1: Optional[float] = None
    lambda2: Optional[float] = None
    gamma: Optional[float] = None


# Making hf config compatible with hydra
HydraCompatCustomizedDPOConfig = builds(
    CustomizedDPOConfig,
    output_dir="${hydra:runtime.output_dir}",
    deepspeed=just(dict()),  # Including deepspeed config in yaml config of Hydra
    populate_full_signature=True,
    hydra_convert="all"
)


class CustomizedDPOTrainer(DPOTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        args: CustomizedDPOConfig = kwargs["args"]

        # Hyperparameters
        self.alpha = args.alpha
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        self.lambda_ = args.lambda_
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

        # Set a default value for gamma
        if args.gamma is None:
            self.gamma = 0.0
        else:
            self.gamma = args.gamma

        self.warm_simpo_reference_free_once = True

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if (
                "reference_chosen_logps" in batch
                and "reference_rejected_logps" in batch
                and self.args.rpo_alpha is not None
        ):
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        # >>>>> Modification: Get chosen and rejected length >>>>>
        @torch.no_grad()
        def get_token_length(labels, attention_mask):
            answer_mask = (labels != self.label_pad_token_id) * attention_mask
            answer_length = answer_mask.sum(dim=-1)
            return answer_length

        chosen_length = get_token_length(batch["chosen_labels"], batch["chosen_attention_mask"])
        rejected_length = get_token_length(batch["rejected_labels"], batch["rejected_attention_mask"])
        # <<<<< Modification: Get chosen and rejected length <<<<<

        # >>>>> Modification: Add more return values >>>>>
        losses, chosen_rewards, rejected_rewards, policy_margin, reference_margin, regularization_term = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_length,
            rejected_length
        )
        # <<<<< Modification: Add more return values <<<<<
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses * self.args.rpo_alpha - policy_chosen_logps_avg

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        # >>>>> Modification: Add new metrics, print to log for debugging >>>>>
        metrics[f"{prefix}regularization_term"] = regularization_term.detach().mean().cpu() \
            if isinstance(regularization_term, torch.Tensor) else float(regularization_term)
        metrics[f"{prefix}length/chosen"] = chosen_length.float().detach().mean().cpu()
        metrics[f"{prefix}length/rejected"] = rejected_length.float().detach().mean().cpu()
        metrics[f"{prefix}margin/policy"] = policy_margin.detach().mean().cpu()
        metrics[f"{prefix}margin/reference"] = reference_margin.detach().mean().cpu()

        logger.debug("step: %s, metrics: %s", self.state.global_step, metrics)
        # <<<<< Modification: Add new metrics, print to log for debugging <<<<<

        return losses.mean(), metrics

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            # >>>>> Modification: Add length to args >>>>>
            chosen_length: torch.LongTensor,
            rejected_length: torch.LongTensor,
            # <<<<< Modification: Add length to args <<<<<
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        # >>>>> Modification: Add additional regularization term to logits >>>>>
        if self.loss_type == "simpo":
            if not self.reference_free and self.warm_simpo_reference_free_once:
                self.warm_simpo_reference_free_once = False
                logger.warning("SimPO is reference-free, but 'args.reference_free' is set to False.")
            a = policy_chosen_logps.to(self.accelerator.device) / chosen_length
            b = policy_rejected_logps.to(self.accelerator.device) / rejected_length
            logits = a - b
            reg_term = - self.gamma
        elif self.loss_type in ["alpha_dpo", "aipo"]:
            assert not self.reference_free, "alpha-DPO is not compatible with reference_free"
            a = policy_chosen_logps - policy_rejected_logps
            b = (reference_chosen_logps - reference_rejected_logps) * (1 + self.alpha)
            logits = a - b
            reg_term = -self.beta * self.gamma
        else:
            reg_term = -self.beta * self.gamma
        # Add margin term
        # Note: Since logits are multiplied by beta in subsequent calculations, but the gamma term is not multiplied
        # by beta by default, we preemptively divide the gamma term by beta here to offset this.
        logits = logits + reg_term / self.beta  # -log_sigmoid(logits * self.beta)
        # <<<<< Modification: Add additional regularization term to logits <<<<<

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        # >>>>> Modification: Add new loss type >>>>>
        if self.loss_type == "sigmoid" or self.loss_type in ["simpo", "alpha_dpo", "aipo"]:
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "nll_loss":
            losses = (
                             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                             - F.logsigmoid(-self.beta * logits) * self.label_smoothing
                     ) - self.alpha * policy_chosen_logps / chosen_length
        elif self.loss_type in ["aipo"]:
            losses = (
                             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                             - F.logsigmoid(-self.beta * logits) * self.label_smoothing
                     ) - self.lambda_ * policy_chosen_logps / chosen_length
        # <<<<< Modification: Add new loss type <<<<<
        elif self.loss_type == "robust":
            losses = (
                             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                             + F.logsigmoid(-self.beta * logits) * self.label_smoothing
                     ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        elif self.loss_type == "sppo_hard":
            # In the paper (https://arxiv.org/pdf/2405.00675), SPPO employs a soft probability approach, estimated using the PairRM score. The probability calculation is conducted outside of the trainer class. The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is set to 1 for the winner and 0 for the loser.
            a = policy_chosen_logps - reference_chosen_logps
            b = policy_rejected_logps - reference_rejected_logps

            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2
        elif self.loss_type == "nca_pair":
            chosen_rewards = (policy_chosen_logps - reference_chosen_logps) * self.beta
            rejected_rewards = (policy_rejected_logps - reference_rejected_logps) * self.beta
            losses = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}.")

        chosen_rewards = (
                self.beta
                * (
                        policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(
                    self.accelerator.device)
                ).detach()
        )
        rejected_rewards = (
                self.beta
                * (
                        policy_rejected_logps.to(self.accelerator.device)
                        - reference_rejected_logps.to(self.accelerator.device)
                ).detach()
        )

        # >>>>> Modification: add policy_margin and reference_margin as metric >>>>>
        policy_margin = (
                self.beta
                * (
                        policy_chosen_logps.to(self.accelerator.device)
                        - policy_rejected_logps.to(self.accelerator.device)
                ).detach()
        )
        if self.reference_free:
            reference_margin = torch.tensor([0], dtype=policy_margin.dtype, device=policy_margin.device)
        else:
            reference_margin = (
                    self.beta
                    * (
                            reference_chosen_logps.to(self.accelerator.device)
                            - reference_rejected_logps.to(self.accelerator.device)
                    ).detach()
            )
        # <<<<< Modification: add policy_margin and reference_margin as metric <<<<<

        # >>>>> Modification: Add return >>>>>
        return losses, chosen_rewards, rejected_rewards, policy_margin, reference_margin, reg_term
        # <<<<< Modification: Add return <<<<<
