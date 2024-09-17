# -*- coding: utf-8 -*-
# @Time    : 5/22/24
# @Author  : Yaojie Shen
# @Project : AIPO
# @File    : ${FILE_NAME}

"""
Self Instruct Creation.
https://aclanthology.org/2023.acl-long.754/
"""

import itertools
import logging
import math
import os
import random
import re
import string
import time
from collections import defaultdict
from dataclasses import field
from typing import Optional, List

import datasets
import numpy as np
import ray
import tiktoken
import torch
from hydra.utils import instantiate
from mm_video.config import runner_store
from mm_video.utils.common import chunk
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from alignment.runner.stages.base import GenerateBaseStage, ModelConfig, GenerateConfig
from alignment.runner.utils.common import save_sharded_dataset, retry_load_sharded_dataset, retry_load_from_disk

logger = logging.getLogger(__name__)

__all__ = ["SelfInstructStage"]


@ray.remote(num_gpus=1)
class BERTActor:
    def __init__(self, model_name='bert-large-uncased', batch_size=16):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()
        self.batch_size = batch_size

    def get_bert_embeddings(self, prompts) -> List[np.ndarray]:
        embeddings = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            inputs = self.tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use attention mask to handle padding
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
            sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1)
            mean_embeddings = sum_embeddings / torch.sum(attention_mask, dim=1)
            embeddings.extend(mean_embeddings.cpu().detach().numpy())
        return embeddings


@ray.remote(num_gpus=1)
class SentenceTransformerActor:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', batch_size=16):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def get_sentence_transformer_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True
        )
        return embeddings


@runner_store(stage="self_instruct", output_dir="${hydra:runtime.output_dir}")
class SelfInstructStage(GenerateBaseStage):
    """
    Self-Instruct: Aligning Language Models with Self-Generated Instructions
    https://aclanthology.org/2023.acl-long.754
    """

    def __init__(
            self,
            iteration: int,
            stage: str,
            output_dir: Optional[str],
            resume: bool = False,
            # Self-instruct creation config
            clustering: bool = False,
            n_instructions_to_generate_total: int = 20000,
            embedding_type: str = "tf-idf",
            sentence_transformer_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
            n_clusters: int = 128,  # K-means n_clusters
            n_instructions_to_generate_per_cluster: int = 64,
            mini_group_size: int = 1,
            n_prompt_instructions: int = 8,
            n_machine_instructions: int = 0,
            max_n_generate_instructions: int = 1,
            kmeans_max_iter: int = 2000,
            prompt_min_length: int = 4,
            prompt_max_length: int = 512,
            additional_filtering: bool = True,
            # Config nodes
            model: ModelConfig = field(default=ModelConfig),
            generate: GenerateConfig = field(default=GenerateConfig),
    ):
        super().__init__(
            iteration=iteration,
            stage=stage,
            output_dir=output_dir,
            model=model,
            generate=generate,
            resume=resume
        )
        self.clustering = clustering
        self.n_instructions_to_generate_total = n_instructions_to_generate_total
        self.embedding_type = embedding_type
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.n_clusters = n_clusters
        self.n_instructions_to_generate_per_cluster = n_instructions_to_generate_per_cluster
        self.mini_group_size = mini_group_size
        self.n_prompt_instructions = n_prompt_instructions
        self.n_machine_instructions = n_machine_instructions
        assert n_machine_instructions <= n_prompt_instructions
        self.max_idx = str(max_n_generate_instructions + n_prompt_instructions)
        self.kmeans_max_iter = kmeans_max_iter
        self.prompt_min_length = prompt_min_length
        self.prompt_max_length = prompt_max_length
        self.additional_filtering = additional_filtering

        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def num_tokens_from_string(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoder.encode(text))
        return num_tokens

    @staticmethod
    def k_means_clustering(sentences: List[str], k: int, max_iter: int) -> List[int]:
        """
        https://www.geeksforgeeks.org/clustering-text-documents-using-k-means-in-scikit-learn/
        :param sentences: list of sentences (string)
        :param k: K-means n_clusters
        :param max_iter: K-means max_iter
        :return: cluster id for the input sentences
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        # vectorizer the text documents
        vectorized_documents = vectorizer.fit_transform(sentences)
        # cluster the documents using k-means
        kmeans = KMeans(n_clusters=k, n_init="auto", max_iter=max_iter, random_state=0)
        kmeans.fit(vectorized_documents)
        return list(kmeans.labels_)

    @staticmethod
    def bert_k_means_clustering(sentences: List[str], k: int, max_iter: int = 300, batch_size: int = 16) -> List[int]:
        """
        Cluster text documents using BERT embeddings and K-Means clustering.

        :param sentences: List of sentences (string)
        :param k: K-Means n_clusters
        :param max_iter: K-Means max_iter
        :param batch_size: Batch size for BERT embedding extraction
        :return: Cluster ids for the input sentences
        """

        num_gpus = torch.cuda.device_count()
        # Create BERT actors in an Actor Pool
        actors = [BERTActor.remote(batch_size=batch_size) for _ in range(num_gpus)]
        actor_pool = ray.util.ActorPool(actors)

        # Extract embeddings in parallel
        print("Calculating embeddings...")
        chunked_prompts = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        embeddings = list(tqdm(itertools.chain.from_iterable(
            actor_pool.map(lambda actor, prompts: actor.get_bert_embeddings.remote(prompts), chunked_prompts)
        ), total=len(sentences)))

        # Cluster the documents using K-Means
        kmeans = KMeans(n_clusters=k, n_init="auto", max_iter=max_iter, random_state=0)
        kmeans.fit(embeddings)
        return list(kmeans.labels_)

    @staticmethod
    def sentence_transformer_k_means_clustering(
            sentences: List[str], k: int, max_iter: int = 300,
            model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
            batch_size: int = 16
    ) -> List[int]:
        """
        Cluster text documents using Sentence Transformer embeddings and K-Means clustering.

        :param sentences: List of sentences (string)
        :param k: K-Means n_clusters
        :param max_iter: K-Means max_iter
        :param model_name: Name of the Sentence Transformer model
        :param batch_size: Batch size for Sentence Transformer embedding extraction
        :return: Cluster ids for the input sentences
        """

        num_gpus = torch.cuda.device_count()
        # Create Sentence Transformer actors in an Actor Pool
        actors = [SentenceTransformerActor.remote(model_name=model_name, batch_size=batch_size) for _ in
                  range(num_gpus)]
        actor_pool = ray.util.ActorPool(actors)

        # Extract embeddings in parallel
        print("Calculating embeddings...")
        chunked_prompts = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        embeddings = list(tqdm(itertools.chain.from_iterable(
            actor_pool.map(lambda actor, prompts: actor.get_sentence_transformer_embeddings.remote(prompts),
                           chunked_prompts)
        ), total=len(sentences)))

        # Cluster the documents using K-Means
        kmeans = KMeans(n_clusters=k, n_init="auto", max_iter=max_iter, random_state=0)
        kmeans.fit(embeddings)
        return list(kmeans.labels_)

    @staticmethod
    def encode_prompt(prompt_instructions: List[str]) -> str:
        """
        Based on:
        https://github.com/yizhongw/self-instruct/blob/0b26ccaa415992100fa32df62d41b994cf928e23/self_instruct/bootstrap_instructions.py#L19
        :param prompt_instructions:
        :return:
        """
        prompt = "Come up with a series of tasks:\n\n"
        for idx, instruction in enumerate(prompt_instructions):
            instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
            prompt += f"{idx + 1}. {instruction}\n"
        prompt += f"{len(prompt_instructions) + 1}."
        return prompt

    def post_process_response(self, response: str) -> List[str]:
        """
        Based on:
        https://github.com/yizhongw/self-instruct/blob/0b26ccaa415992100fa32df62d41b994cf928e23/self_instruct/bootstrap_instructions.py#L41
        :param response:
        :return:
        """

        def find_word_in_string(w, s):
            return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)

        if not response:
            return []

        raw_instructions = re.split(r"\n\d+\s?\. ", response)
        instructions = []
        for inst in raw_instructions:
            inst = re.sub(r"\s+", " ", inst).strip()
            inst = inst.strip().capitalize()
            if inst == "":
                continue
            # filter out too short or too long instructions
            n_tokens = self.num_tokens_from_string(inst)
            if n_tokens < self.prompt_min_length or n_tokens > self.prompt_max_length:
                continue
            # filter based on keywords that are not suitable for language models.
            if any(find_word_in_string(word, inst) for word in
                   ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw",
                    "plot", "go to"]):
                continue
            # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot
            # of such instructions.
            # And it's a bit confusing whether the model need to write a program or directly output the result.
            # Here we filter them out.
            # Note this is not a comprehensive filtering for all programming instructions.
            if inst.startswith("Write a program"):
                continue
            # filter those starting with punctuation
            if inst[0] in string.punctuation:
                continue
            # filter those starting with non-english character
            if not inst[0].isascii():
                continue

            if self.additional_filtering:
                if inst.startswith("Detailed instructions:"):
                    continue
                if inst.startswith("Task definition:"):
                    continue
                # Check for filter keywords and truncate if found
                if "example:" not in inst.lower():
                    for keyword in ["answer:", "output:", "solution:", "a:"]:
                        keyword_index = inst.lower().find(keyword.lower())
                        if keyword_index != -1:
                            inst = inst[:keyword_index].strip()

            instructions.append(inst)
        return instructions

    def generate_self_instruct_pipeline(self, prompt_dataset: datasets.Dataset) -> datasets.Dataset:
        print_example_once = True
        logger.info("Generating self instructs...")
        assert "prompt" in list(prompt_dataset.features)

        if self.clustering:
            dataset_path = os.path.join(self.output_dir, "prompt_dataset_clustered")
            prompts = prompt_dataset[:]["prompt"]
            # Since we do not know how long it will take to run K-means clustering, so we run it on all shards so that they
            # will finish at similar time points.
            logger.info("K-means clustering...")
            s_time = time.time()
            if self.embedding_type == "tf-idf":
                cluster_id = self.k_means_clustering(prompts, k=self.n_clusters, max_iter=self.kmeans_max_iter)
            elif self.embedding_type == "bert":
                cluster_id = self.bert_k_means_clustering(prompts, k=self.n_clusters, max_iter=self.kmeans_max_iter)
            elif self.embedding_type == "sentence_transformers":
                cluster_id = self.sentence_transformer_k_means_clustering(
                    prompts,
                    k=self.n_clusters,
                    max_iter=self.kmeans_max_iter,
                    model_name=self.sentence_transformer_model_name,
                )
            else:
                raise ValueError(f"Unknown embedding type: {self.embedding_type}")
            logger.info("K-means clustering completed in {:.2f} seconds".format(time.time() - s_time))
            if self.generate_cfg.shard_id == 0:
                # Only use the clustering results from shard 0
                prompt_dataset = prompt_dataset.add_column(name="cluster_id", column=cluster_id)
                prompt_dataset.save_to_disk(dataset_path)
            else:
                prompt_dataset = retry_load_from_disk(dataset_path, retry=60, wait_interval=30)

            # Group by cluster_id
            logger.info("Grouping by cluster id...")
            cluster_id_to_prompt_pool = defaultdict(list)
            for example in tqdm(prompt_dataset):
                cluster_id_to_prompt_pool[example["cluster_id"]].append(example["prompt"])
            cluster_id_to_prompt_pool = dict(cluster_id_to_prompt_pool)

            # Select a group of cluster_id.
            # Each shard processing a group of cluster_id, generate N prompts for each cluster.
            processing_cluster_id = list(range(self.n_clusters))
            processing_cluster_id = chunk(
                processing_cluster_id,
                n_chunks=self.generate_cfg.num_shards
            )[self.generate_cfg.shard_id]

            n_instructions_to_generate_per_cluster = self.n_instructions_to_generate_per_cluster
        else:
            # Build fake clustering results, set number of instructions to generate on each shard
            cluster_id_to_prompt_pool = {0: prompt_dataset[:]["prompt"]}
            processing_cluster_id = [0]
            n_instructions_to_generate_per_cluster = (self.n_instructions_to_generate_total //
                                                      self.generate_cfg.num_shards)

        # Generate
        # 1. Build N (=self.mini_group_size) zero-shot prompts for each cluster.
        # 2. Generate for all prompts together (for best performance).
        # 3. Parse responses, adding a new prompt to each cluster.
        # 4. Identify the clusters whose prompts have not yet reached the target number.
        # 5. Go back to step 1 and repeat the above steps until prompts for all clusters have reached the target number.
        generator = self._build_generator()
        new_instructions = defaultdict(list)  # cluster id, instruction
        progress = tqdm(desc="Generating self-instructs", dynamic_ncols=True,
                        total=n_instructions_to_generate_per_cluster * len(processing_cluster_id))

        for _ in range(math.ceil(n_instructions_to_generate_per_cluster / self.mini_group_size * 2)):
            remaining_clusters = [
                cluster_id for cluster_id in processing_cluster_id
                if len(new_instructions[cluster_id]) < n_instructions_to_generate_per_cluster
            ]
            if not remaining_clusters:
                break

            batch_prompts = []
            cluster_prompt_map = []
            for cluster_id in remaining_clusters:
                cluster_prompt_pool = cluster_id_to_prompt_pool[cluster_id]
                cluster_instructions = new_instructions[cluster_id]

                for _ in range(self.mini_group_size):
                    # Sample some prompts from the pool
                    prompt_instructions = random.sample(
                        cluster_instructions,
                        k=min(self.n_machine_instructions, len(cluster_instructions))
                    )
                    prompt_instructions += random.sample(
                        cluster_prompt_pool,
                        k=min(len(cluster_prompt_pool), self.n_prompt_instructions - len(prompt_instructions))
                    )
                    batch_prompts.append(self.encode_prompt(prompt_instructions))
                    cluster_prompt_map.append(cluster_id)

            if self.generate_cfg.vllm:
                results = generator.generate(
                    batch_prompts,
                    stop=["\n\n", f"\n{self.max_idx}", f"{self.max_idx}.", f"{self.max_idx} ."],
                    generate_kwargs=dict(use_tqdm=False)  # Avoid verbose outputs
                )
            else:
                results = generator.generate(
                    batch_prompts,
                    stop_strings=["\n\n", f"\n{self.max_idx}", f"{self.max_idx}.", f"{self.max_idx} ."],
                    tokenizer=self.tokenizer
                )

            if print_example_once and self.generate_cfg.shard_id == 0:
                print_example_once = False
                logger.info("Self instruct prompt example:\n%s", batch_prompts[0])
                logger.info("Self instruct output example:\n%s", results[0])

            for cluster_id, result in zip(cluster_prompt_map, results):
                new_responses = self.post_process_response(result)
                progress.update(
                    min(len(new_responses),
                        max(0, n_instructions_to_generate_per_cluster - len(new_instructions[cluster_id])))
                )
                new_instructions[cluster_id].extend(new_responses)
                # Limit the new instruction for each cluster to 'n_instructions_to_generate_per_cluster'
                new_instructions[cluster_id] = new_instructions[cluster_id][
                                               :n_instructions_to_generate_per_cluster]

        progress.close()

        # Create dataset based on new instructions
        dataset = [
            {
                "prompt": instruction,
                "messages": [{"role": "user", "content": instruction}],
                "metadata": {
                    "cluster_id": cluster_id
                }
            } for cluster_id, instructions in new_instructions.items() for instruction in instructions
        ]
        return datasets.Dataset.from_list(dataset)

    def _run(self, cfg: DictConfig):
        prompt_dataset = instantiate(cfg.dataset)
        # Get prompt from the chat dataset
        new_instruction_dataset = self.generate_self_instruct_pipeline(prompt_dataset)
        save_sharded_dataset(dataset=new_instruction_dataset,
                             dataset_root=os.path.join(self.output_dir, "self_instruct"),
                             shard_id=self.generate_cfg.shard_id, n_shards=self.generate_cfg.num_shards)

    def resume_check(self):
        retry_load_sharded_dataset(os.path.join(self.output_dir, "self_instruct"), retry=1, wait_interval=1)
