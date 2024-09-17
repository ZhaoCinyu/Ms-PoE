import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils.modify_arch.mspoe_models import MsPoELlamaForCausalLM, MsPoEQwen2ForCausalLM, MsPoEMistralForCausalLM
from bayes_opt import BayesianOptimization

def setup_models(args, attn_implementation="flash_attention_2"):
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

    if args.enable_ms_poe:
        print('Using Ms-PoE Positional Embedding')
        config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
        config.compress_ratio_min = args.compress_ratio_min
        config.compress_ratio_max = args.compress_ratio_max
        config.head_type = args.head_type
        print('Compress Ratio: from {} to {}'.format(config.compress_ratio_min, config.compress_ratio_max))
        if "mistral" in args.model_name.lower():
            Model = MsPoEMistralForCausalLM
        elif "qwen" in args.model_name.lower():
            Model = MsPoEQwen2ForCausalLM
        else:
            Model = MsPoELlamaForCausalLM
        model = Model.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir, device_map="auto",
                                      torch_dtype="auto", attn_implementation=attn_implementation)
    else:
        print('Using the Baseline Model')
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, device_map="auto",
                                                     torch_dtype="auto", attn_implementation=attn_implementation)

    return config, tokenizer, model

def evaluate_model(compress_ratio_min, compress_ratio_max):
    if compress_ratio_max <= compress_ratio_min + 0.4:
        return float('-inf')  # Invalid configuration

    class Args:
        model_name = 'your-model-name'
        cache_dir = 'your-cache-dir'
        enable_ms_poe = True
        apply_layers = "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
        compress_ratio_min = compress_ratio_min
        compress_ratio_max = compress_ratio_max
        head_type = 'your-head-type'

    args = Args()
    config, tokenizer, model = setup_models(args)
    
    # Here you should implement your evaluation logic, e.g., calculating the loss on a validation set
    # For demonstration, we return a dummy value
    validation_loss = 0.0  # Replace with actual evaluation logic
    return -validation_loss  # Bayesian optimization minimizes the function

# Define the parameter bounds
pbounds = {
    'compress_ratio_min': (1, 3),
    'compress_ratio_max': (1, 3)
}

optimizer = BayesianOptimization(
    f=evaluate_model,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=10,
)

print(optimizer.max)