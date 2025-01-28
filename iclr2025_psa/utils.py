from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase as Tokenizer
from transformers import PreTrainedModel as Model
from dataclasses import dataclass
from typing import Iterable

from torch.distributions import Laplace
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model_and_tokenizer(model_name: str, load_bfloat=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Note: you must have installed 'accelerate', 'bitsandbytes' to load in 8bit
    if load_bfloat:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
        )
    model = model.to(device)
    return model, tokenizer

B_INST = "[INST]"
E_INST = "[/INST]"
SYS_PROMPT = "You are a helpful, honest assistant."
B_SYS = "<<SYS>>"
E_SYS = "<</SYS>>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
# Note: No EOS token is used in original codebase
prompt_template = f"{BOS_TOKEN} {B_INST} \n{B_SYS}{SYS_PROMPT}{E_SYS}\n\n {{question}} {E_INST} {{answer}}"

MWEData = list[dict[str, str]]


def make_pos_neg_pair(mwe_data: MWEData, tokenizer) -> tuple[str, str]:
    """ Creates a (positive, negative) pair for getting contrastive activations """
    if 'llama' or 'mistral' in tokenizer.name_or_path.lower():
        pos = prompt_template.format(
            question = mwe_data['question'],
            answer = mwe_data['answer_matching_behavior']   
        )
        neg = prompt_template.format(
            question = mwe_data['question'],
            answer = mwe_data['answer_not_matching_behavior']
        )
    elif 'qwen' in tokenizer.name_or_path.lower():
        end_token = '<|im_end|>'
        pos = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a helpful, honest assistant.'},
            {'role': 'user', 'content': mwe_data['question']},
            {'role': 'assistant', 'content': mwe_data['answer_matching_behavior']}
        ], tokenize=False, add_generation_prompt=False)
        idx = pos.rfind(end_token)
        pos = pos[:idx] 
        neg = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a helpful, honest assistant.'},
            {'role': 'user', 'content': mwe_data['question']},
            {'role': 'assistant', 'content': mwe_data['answer_not_matching_behavior']}
        ], tokenize=False, add_generation_prompt=False)
        idx = neg.rfind(end_token)
        neg = neg[:idx] 

    elif 'gemma' in tokenizer.name_or_path.lower():
        end_token = '<end_of_turn>'
        pos = tokenizer.apply_chat_template([
            # {'role': 'system', 'content': 'You are a helpful, honest assistant.'},
            {'role': 'user', 'content': mwe_data['question']},
            {'role': 'assistant', 'content': mwe_data['answer_matching_behavior']}
        ], tokenize=False, add_generation_prompt=False)
        idx = pos.rfind(end_token)
        pos = pos[:idx] 
        neg = tokenizer.apply_chat_template([
            # {'role': 'system', 'content': 'You are a helpful, honest assistant.'},
            {'role': 'user', 'content': mwe_data['question']},
            {'role': 'assistant', 'content': mwe_data['answer_not_matching_behavior']}
        ], tokenize=False, add_generation_prompt=False)
        idx = neg.rfind(end_token)
        neg = neg[:idx] 
    return pos, neg

def make_dataset(list_mwe_data: list[MWEData], tokenizer) -> list[tuple[str, str]]:
    """ Creates a list of (positive, negative) pairs for getting contrastive activations """
    return [make_pos_neg_pair(mwe_data, tokenizer) for mwe_data in list_mwe_data]

def get_probabilities(logprobs: list[float]) -> list[float]:
    """ Converts log-probabilities to a normalized probability distribution """
    min_logprob = min(logprobs)
    # Shift the range to avoid underflow when exponentiating
    logprobs = [logprob - min_logprob for logprob in logprobs]
    # Exponentiate and normalize
    probs = [math.exp(logprob) for logprob in logprobs]
    total = sum(probs)
    probs = [prob / total for prob in probs]
    return probs

@dataclass
class TokenProb:
    token_id: int
    logprob: float
    text: str

@dataclass
class TextProbs:
    text: str
    token_probs: list[TokenProb]

    @property
    def sum_logprobs(self) -> float:
        return sum([tp.logprob for tp in self.token_probs])

    def __repr__(self) -> str:
        return f"TextProbs({self.text}:{self.sum_logprobs:.2f})"
    

def get_text_probs(input: str, model: Model, tokenizer: Tokenizer, ) -> TextProbs:
    """ Get the token-wise probabilities of a given input """
    inputs = tokenizer(input, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=False, return_dict=True)
    logprobs = torch.log_softmax(outputs.logits, dim=-1)
    # .detach().cpu()
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    logprobs = logprobs[:, :-1, :]
    target_ids = inputs.input_ids[:, 1:]
    # Get the probability of the subsequent token
    # target_ids = target_ids.detach().cpu()
    gen_logprobs = torch.gather(logprobs, 2, target_ids[:, :, None]).squeeze(-1)[0]

    text_logprobs: list[TokenProb] = []
    for token, p in zip(target_ids[0], gen_logprobs):
        if token not in tokenizer.all_special_ids:
            text_logprobs.append(
                TokenProb(
                    token_id=token.item(),
                    text=tokenizer.decode(token),
                    logprob=p.item(),
                )
            )
    return TextProbs(text=input, token_probs=text_logprobs)


def evaluate_model(
    model: Model, 
    tokenizer: Tokenizer, 
    dataset: Iterable[tuple[str, str]],
    show_progress: bool = False
):
    """ Evaluate model on dataset and return normalized probability of correct answer """
    total_pos_prob = 0.0
    for pos_prompt, neg_prompt in tqdm(dataset, disable=not show_progress, desc="Evaluating"):
        pos: TextProbs = get_text_probs(pos_prompt, model, tokenizer)
        neg: TextProbs = get_text_probs(neg_prompt, model, tokenizer)
        # NOTE: We compare logprobs of the full (prompt + response).  
        # This is equivalent to comparing response log-probs only.  
        # Because the prompts are the same for both positive and negative, 
        # the prompt log-probs factor out as an additive constant in the total log-probs.
        # and so the relative difference in log-probs is unchanged.
        pos_prob, _ = get_probabilities([pos.sum_logprobs, neg.sum_logprobs])
        total_pos_prob += pos_prob
    return total_pos_prob / len(dataset)


def generate_text(prompt, model, tokenizer, temperature=0.6, max_length=50, do_sample=True):
    inps = tokenizer(prompt, return_tensors='pt').to(device)
    outs = model.generate(**inps, temperature=temperature, max_length=max_length, do_sample=do_sample)
    print(tokenizer.decode(outs[0]))

def generate_text_with_template(prompt, model, tokenizer, temperature=0.6, max_new_tokens=50, do_sample=True):
    prompt_dict = [
        {
            "role": "user", "content": prompt 
        }
    ]
    inps = tokenizer.apply_chat_template(prompt_dict, return_tensors='pt', add_generation_prompt=True).to(device)
    outs = model.generate(inps, temperature=temperature, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return tokenizer.decode(outs[0])