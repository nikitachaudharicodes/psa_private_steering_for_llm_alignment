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
# from huggingface_hub import login 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MWEData = list[dict[str, str]]

def get_results(dataset, prompt=None, pca=False, layers=[11,12,13,14,15], noise_multiplier=0.01, clip=20):
    train_data: list[MWEData] = json.load(open(f"./datasets/generate/{dataset}/generate_dataset.json", 'r'))
    test_data: list[MWEData] = json.load(open(f"./datasets/test/{dataset}/test_dataset_ab.json", 'r'))

    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(test_data)
    train_data = train_data
    test_data = test_data

    train_dataset = make_dataset(train_data, tokenizer)
    test_dataset = make_dataset(test_data, tokenizer)


    # def fisher_discriminant(pos, neg):
    #     mu_pos = pos.mean(dim=0)
    #     mu_neg = neg.mean(dim=0)
    #     pos_centered = pos - mu_pos
    #     neg_centered = neg - mu_neg

    #     cov_pos = pos_centered.T @ pos_centered / (pos.shape[0] - 1)
    #     cov_neg = neg_centered.T @ neg_centered / (neg.shape[0] - 1)
    #     cov = cov_pos + cov_neg + 1e-4 * torch.eye(pos.shape[1], device=pos.device)

    #     direction = torch.linalg.solve(cov, mu_pos - mu_neg)
    #     return direction / torch.norm(direction)


    def private_fisher_discriminant(pos, neg, clip=20, noise_multiplier=0.02):
        C = clip
        d = pos.shape[1]
        device = pos.device

        def clip_rows(mat):
            norms = torch.norm(mat, dim=1, keepdim=True)
            scale = torch.clamp(C / norms, max=1.0)
            return mat * scale

        pos_clipped = clip_rows(pos)
        neg_clipped = clip_rows(neg)

        mu_pos = pos_clipped.mean(dim=0) + noise_multiplier * torch.normal(0, 1.0, size=(d,), device=device)
        mu_neg = neg_clipped.mean(dim=0) + noise_multiplier * torch.normal(0, 1.0, size=(d,), device=device)

        pos_centered = pos_clipped - mu_pos
        neg_centered = neg_clipped - mu_neg

        cov_pos = pos_centered.T @ pos_centered / (pos.shape[0] - 1)
        cov_neg = neg_centered.T @ neg_centered / (neg.shape[0] - 1)
        cov = cov_pos + cov_neg + 1e-4 * torch.eye(d, device=device)

        direction = torch.linalg.solve(cov, mu_pos - mu_neg)
        return direction / torch.norm(direction)


    # def scaled_mean(pos, neg):
    #     diff = pos - neg
    #     scale = torch.max(torch.norm(diff, dim=1))
    #     diff /= scale
    #     return torch.mean(diff, dim=0)

    
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

    # private_steering = train_steering_vector(
    #     model,
    #     tokenizer,
    #     train_dataset,
    #     read_token_index=-2,
    #     show_progress=True,
    #     aggregator=priv_mean,
    #     layers=layers
    # )

    # print("Mean Steering")
    # for multiplier in (-2, 0, 2):
    #     with mean_steering.apply(model, multiplier=multiplier, min_token_index=0):
    #         result = evaluate_model(model, tokenizer, test_dataset)
    #         print(f"{multiplier=} | {result=}")
    #         # generate_text(prompt, model, tokenizer)

    # print("Private Mean Steering")
    # for multiplier in (-2, 2):
    #     with private_steering.apply(model, multiplier=multiplier, min_token_index=0):
    #         result = evaluate_model(model, tokenizer, test_dataset)
    #         print(f"{multiplier=} | {result=}")
    #         if prompt:
    #             generate_text(prompt, model, tokenizer)

    # print("Fisher Discriminant Steering")
    # fisher_steering = train_steering_vector(
    #     model,
    #     tokenizer,
    #     train_dataset,
    #     read_token_index=-2,
    #     show_progress=True,
    #     aggregator=fisher_discriminant,
    #     layers=layers
    # )
    # for multiplier in (-2, 0, 2):
    #     with fisher_steering.apply(model, multiplier=multiplier, min_token_index=0):
    #         result = evaluate_model(model, tokenizer, test_dataset)
    #         print(f"Fisher | {multiplier=} | {result=}")

    print("Private Fisher Discriminant Steering")
    private_fisher_steering = train_steering_vector(
        model,
        tokenizer,
        train_dataset,
        read_token_index=-2,
        show_progress=True,
        aggregator=lambda pos, neg: private_fisher_discriminant(pos, neg, clip=clip, noise_multiplier=noise_multiplier),
        layers=layers
    )
    for multiplier in (-2, 0, 2):
        with private_fisher_steering.apply(model, multiplier=multiplier, min_token_index=0):
            result = evaluate_model(model, tokenizer, test_dataset)
            print(f"Private Fisher | {multiplier=} | {result=}")

    


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
    clip=args.clip,
    pca=False
)