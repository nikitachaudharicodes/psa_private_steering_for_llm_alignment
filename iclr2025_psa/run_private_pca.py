from utils import * 

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json 
import random
from torch.distributions import Laplace
import math
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase as Tokenizer
from transformers import PreTrainedModel as Model
from dataclasses import dataclass
from typing import Iterable
from steering_vectors import train_steering_vector, pca_aggregator
import matplotlib.pyplot as plt
import argparse 
from autodp.mechanism_zoo import GaussianMechanism
from autodp.calibrator_zoo import eps_delta_calibrator
import numpy as np


from huggingface_hub import login

login(token="")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

MWEData = list[dict[str, str]]

def get_results(dataset, prompt=None, pca=False, layers=[11,12,13,14,15], noise_multiplier=0.01, clip=10):
    train_data: list[MWEData] = json.load(open(f"./datasets/generate/{dataset}/generate_dataset.json", 'r'))
    test_data: list[MWEData] = json.load(open(f"./datasets/test/{dataset}/test_dataset_ab.json", 'r'))

    random.seed(42)
    random.shuffle(train_data)
    # train_data = train_data[:10]
    random.shuffle(test_data)
    train_data = train_data
    test_data = test_data

    train_dataset = make_dataset(train_data, tokenizer)
    test_dataset = make_dataset(test_data, tokenizer)


    def scaled_mean(pos, neg):
        diff = pos - neg
        scale = torch.max(torch.norm(diff, dim=1))
        diff /= scale
        return torch.mean(diff, dim=0)

    
    # def priv_mean(pos, neg):
    #     diff = pos - neg
    #     C = clip
    #     norms = torch.norm(diff, dim=1)
    #     scale_factors = torch.clamp(C/norms, max=1.0).view(-1, 1)
    #     diff = diff * scale_factors
    #     diff /= C
    #     mu = torch.mean(diff, dim=0)
    #     noise = torch.normal(0, torch.tensor(1.), size=mu.shape).to(device)
    #     return mu + (noise_multiplier * noise)
    
    # def priv_mean_rdp(pos, neg, alpha=10, epsilon_bar=100.0, delta=1e-5):
    #     diff = pos - neg
    #     C = clip
    #     norms = torch.norm(diff, dim=1)
    #     scale_factors = torch.clamp(C / norms, max=1.0).view(-1, 1)
    #     diff = diff * scale_factors
    #     # diff /= C
    #     mu = torch.mean(diff, dim=0)

    #     # Compute sigma based on RDP
    #     sensitivity = 1.0  # Since we've clipped the norm to 1
    #     sigma = np.sqrt((sensitivity ** 2 * alpha) / (2 * epsilon_bar))
    #     print(f"Sigma: {sigma}")

    #     noise = torch.normal(0, sigma, size=mu.shape).to(device)

    #     print(f"Mean diff norm: {mu.norm().item():.4f}")
    #     print(f"Noise norm: {noise.norm().item():.4f}")

    #     return mu + noise

    def priv_mean_dp_pca(pos, neg, epsilon=1.0, delta=1e-5, clip=10.0, k=1):
        """
        Computes a differentially private mean vector using DP-PCA.

        Parameters:
        - pos: Tensor of positive examples.
        - neg: Tensor of negative examples.
        - epsilon: Privacy parameter epsilon.
        - delta: Privacy parameter delta.
        - clip: Clipping norm for individual differences.
        - k: Number of principal components to retain.

        Returns:
        - A differentially private mean vector.
        """
        # Compute the difference vectors
        diff = pos - neg

        # Clip the norms of the difference vectors
        norms = torch.norm(diff, dim=1, keepdim=True)
        scale = torch.clamp(clip / norms, max=1.0)
        diff = diff * scale

        # Compute the covariance matrix
        cov = torch.matmul(diff.T, diff) / diff.shape[0]

        # Add symmetric Gaussian noise to the covariance matrix
        sensitivity = (2 * clip ** 2) / diff.shape[0]
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = torch.normal(0, sigma, size=cov.shape).to(cov.device)
        noise = (noise + noise.T) / 2  # Ensure the noise is symmetric
        cov_noisy = cov + noise

        # Compute the top-k eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(cov_noisy)
        topk = eigvecs[:, -k:]

        # Project the mean difference onto the top-k principal components
        mu = torch.mean(diff, dim=0)
        mu_proj = torch.matmul(topk, torch.matmul(topk.T, mu))

        return mu_proj


    # if pca:
    #     pca_steering = train_steering_vector(
    #         model,
    #         tokenizer,
    #         train_dataset,
    #         read_token_index=-2,
    #         show_progress=True,
    #         aggregator=pca_aggregator(),
    #         layers=layers
    #     )
    #     print("PCA Steering")
    #     for multiplier in (-2, 2):
    #         with pca_steering.apply(model, multiplier=multiplier, min_token_index=0):
    #             result = evaluate_model(model, tokenizer, test_dataset)
    #             print(f"{multiplier=} | {result=}")
    
    # mean_steering = train_steering_vector(
    #     model, 
    #     tokenizer,
    #     train_dataset,
    #     read_token_index=-2,
    #     show_progress=True,
    #     aggregator=scaled_mean,
    #     layers=layers
    # )

    private_steering = train_steering_vector(
        model,
        tokenizer,
        train_dataset,
        read_token_index=-2,
        show_progress=True,
        aggregator=lambda pos, neg: priv_mean_dp_pca(pos, neg, epsilon=1.0, delta=1e-5, clip=10.0, k=1),
        layers=layers
    )

    # print("Mean Steering")
    # for multiplier in (-2, 0, 2):
    #     with mean_steering.apply(model, multiplier=multiplier, min_token_index=0):
    #         result = evaluate_model(model, tokenizer, test_dataset)
    #         print(f"{multiplier=} | {result=}")
    #         # generate_text(prompt, model, tokenizer)

    print("Private PCA Steering")
    for multiplier in (-2, 2):
        with private_steering.apply(model, multiplier=multiplier, min_token_index=0):
            result = evaluate_model(model, tokenizer, test_dataset)
            print(f"{multiplier=} | {result=}")
            if prompt:
                generate_text(prompt, model, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-2-7B-chat-hf",
        help="huggingface llm identifier"
    )
    parser.add_argument(
        "--dataset",
        default="sycophancy",
        help="name of dataset",
        choices=['sycophancy', 'hallucination', 'refusal', 'myopic-reward', 'survival-instinct', 'coordinate-other-ais', 'corrigible-neutral-HHH']
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        help="layers to apply psa on"
    )
    parser.add_argument(
        "--noise_multiplier",
        default=0.02,
        type=float,
        help="amount of noise added in psa"
    )
    parser.add_argument(
        "--clip",
        default=20,
        type=int,
        help="clipping factor"
    )
    args = parser.parse_args()


model_name = args.model
model, tokenizer = get_model_and_tokenizer(model_name, load_bfloat=False)

get_results(
    dataset=args.dataset,
    layers=args.layers,
    noise_multiplier=args.noise_multiplier,
    pca=True
)