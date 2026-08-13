import os
default_hf_home = os.path.join(os.path.dirname(__file__), 'models')
os.environ.setdefault('HF_HOME', os.environ.get('HF_HOME', default_hf_home))
os.environ.setdefault('WANDB_MODE', os.environ.get('WANDB_MODE', 'offline'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Do not hardcode WANDB API key in source. If set in environment, wandb will use it.
import argparse
import random
import string
import torch
import numpy as np
from loggers import WandBLogger
from judges import load_judge, judge_rule_based
from language_models import GPT
from prompts import *
from conversers import *
from utils import insert_adv_string, schedule_n_to_change_fixed, schedule_n_to_change_prob, extract_logprob, early_stopping_condition
import time
import heapq
# from beam_search import beam_search_attack, beam_search_attack_random, beam_search_attack_stochastic, manipulate_output_token_probability, manipulate_first_token_then_generate
from beam_search import *
from random_search import random_search
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

SCORE = 1  # the score assigned by the judge LLM 

### Define Input Dataset ###
DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / 'results-nlpcc' / 'SIDE+DGLM-PAIR_dataset'
# DEFAULT_SAVE_PATH = Path(__file__).resolve().parent / 'results-nlpcc' / 'SIDE+DGLM-jailbreakbench-PAIR-direction'

PAIR_PATH = Path(__file__).resolve().parent / 'dataset' / 'side' / 'reverse_pair' / 'harmful_harmless_pair.csv'




def main(args):
    ### initialization
    use_manipulate = args.use_manipulate
    use_template = args.use_template
    use_ic = args.use_ic
    target_token = args.target_str
    if use_template:
        print("Using template.")
    if use_manipulate:
        print("Using manipulation of the first few tokens.")
    if use_ic:
        print("Using in-context examples.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    save_path = Path(getattr(args, 'save_path', DEFAULT_SAVE_PATH))
    if getattr(args, 'logits_method', None) == 'PS':
        save_path = save_path / f'PS'
    if getattr(args, 'logits_method', None) == 'DGLM':
        save_path = save_path / f'DGLM/lambda_{int(args.lambda_)}'
    if getattr(args, 'logits_method', None) == 'EP+DGLM':
        save_path = save_path / f'EP+DGLM/lambda_{int(args.lambda_)}'
    if getattr(args, 'logits_method', None) == 'JSF+PGT':
        save_path = save_path / f'JSF+PGT/lambda_{int(args.lambda_)}'
    if getattr(args, 'logits_method', None) == 'JSF+TDS':
        save_path = save_path / f'JSF+TDS/lambda_{int(args.lambda_)}'
    if getattr(args, 'logits_method', None) == 'PGT+TDS':
        save_path = save_path / f'PGT+TDS/lambda_{int(args.lambda_)}'
    if getattr(args, 'logits_method', None) == 'JSF+PGT+PS':
        save_path = save_path / f'JSF+PGT+PS/lambda_{int(args.lambda_)}'
    if getattr(args, 'logits_method', None) == 'JSF+PGT+DGLM':
        save_path = save_path / f'JSF+PGT+DGLM/lambda_{int(args.lambda_)}'
    
    save_path.mkdir(parents=True, exist_ok=True)
    # Initialize models and logger 
    print("attack ", args.target_model)
    targetLM = load_target_model(args)

    tokenizer = targetLM.model.tokenizer

    ###read advbench data###
    final_prompts = []
    init_prompts = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            final_prompts.append(row['final_prompt'])
            init_prompts.append(row['origin_prompt'])
  
    goals = final_prompts
    # goals = init_prompts[:100]
    # init_prompts = init_prompts[:100]

    start_index = max(0, int(getattr(args, 'start_index', 0)))
    if start_index >= len(goals):
        raise ValueError(f'start_index={start_index} is out of range for dataset size {len(goals)}')

    goals = goals[start_index:]
    init_prompts = init_prompts[start_index:]
    
    print("Done loading advbench data.")

    ### 开始攻击 ###
    total = len(goals)
    attack = total
    # attack = 0
    current_all = 0
    final_response_texts = []

    start_time = time.time()
    # Prepare logits-attack resources if requested
    d_vocab = None
    logits_method = getattr(args, 'logits_method', None)
    if use_manipulate:
        # import helpers from main_logits_attack
        from main_logits_attack import load_direction_vector, compute_d_vocab_from_direction, compute_direction_on_the_fly
        direction_dir = Path(args.direction_dir) if hasattr(args, 'direction_dir') and args.direction_dir else Path(__file__).resolve().parent / 'analysis_output' / 'reject_direction' / args.target_model
        if direction_dir.exists() and any(direction_dir.glob('v_layer_*.npy')):
            directions, _ = load_direction_vector(direction_dir)
            # choose layer
            if args.direction_layer and args.direction_layer != 'auto':
                layer = int(args.direction_layer)
                if layer not in directions:
                    raise ValueError(f'Requested direction layer {layer} not found in {direction_dir}')
            else:
                layers_sorted = sorted(directions.keys())
                layer = layers_sorted[-1]
            direction_vec = directions[layer]
        else:
            # compute on the fly from pair file
            pair_file = Path(args.pair_file) if hasattr(args, 'pair_file') and args.pair_file else Path(__file__).resolve().parent / 'dataset' / 'side' / 'reverse_pair' / 'harmful_harmless_pair.csv'
            safe_prompts = []
            unsafe_prompts = []
            with open(pair_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if 'harmless' in row and 'harmful' in row:
                        safe_prompts.append(row['harmless'])
                        unsafe_prompts.append(row['harmful'])
                    if args.direction_limit and i+1 >= args.direction_limit:
                        break
            directions, _ = compute_direction_on_the_fly(targetLM, safe_prompts, unsafe_prompts, layer_indices=None)
            layers_sorted = sorted(directions.keys())
            layer = layers_sorted[-1]
            direction_vec = directions[layer]
        d_vocab = compute_d_vocab_from_direction(targetLM, direction_vec)
        tv = d_vocab.detach().cpu()
        # print top-5 tokens in d_vocab for debugging
        # try:
        #     k = min(20, int(tv.numel()))
        #     top_vals, top_idx = torch.topk(tv.abs(), k=k)
        #     top_tokens = tokenizer.convert_ids_to_tokens(top_idx.tolist())
        #     print('[DGLM] top-10 tokens in d_vocab (by abs value):')
        #     for rank, (tid, tstr, val) in enumerate(zip(top_idx.tolist(), top_tokens, top_vals.tolist()), start=1):
        #         print(f"  {rank}. id={tid} token={tstr!r} value={val:.6f}")
        # except Exception as e:
        #     print('Failed to print top tokens in d_vocab:', e)

        # print specific token's d_vocab value for debugging
        # try:
        #     token_text = target_token if isinstance(target_token, str) else str(target_token)
        #     token_text =  'as'
        #     if hasattr(tokenizer, 'encode'):
        #         token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        #     else:
        #         token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token_text))

        #     if not token_ids:
        #         print(f"[DGLM] No token ids found for token string: {token_text!r}")
        #     else:
        #         if not isinstance(token_ids, (list, tuple)):
        #             token_ids = [token_ids]
        #         print(f"[DGLM] d_vocab values for token string: {token_text!r}")
        #         for tid in token_ids:
        #             if isinstance(tid, torch.Tensor):
        #                 tid = int(tid.item())
        #             if not (0 <= tid < tv.numel()):
        #                 print(f"  token id={tid} out of range (vocab size={tv.numel()})")
        #                 continue
        #             token_repr = tokenizer.convert_ids_to_tokens(tid) if hasattr(tokenizer, 'convert_ids_to_tokens') else str(tid)
        #             print(f"  token id={tid} token={token_repr!r} d_vocab={float(tv[tid].item()):.6f}")
        # except Exception as e:
        #     print('Failed to print specific token d_vocab value:', e)
        
        print(f'Prepared {logits_method} d_vocab from layer {layer}')
        dglm_first_k = getattr(args, 'dglm_first_k', 5)
    output_file = save_path / f'{args.target_model}_{use_manipulate}_{use_template}_{args.logits_method}.csv'
    file_exists = output_file.exists()
    write_header = (not file_exists) or output_file.stat().st_size == 0 or start_index == 0

    with open(output_file, 'a' if file_exists and start_index > 0 else 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['final_prompt', 'output'])
            f.flush()
            os.fsync(f.fileno())

        with tqdm(total=len(goals)) as pbar:
            for goal, init_prompt in zip(goals, init_prompts):
                current_all += 1
                if use_template:
                    orig_msg = goal
                else:
                    orig_msg = init_prompt
                if use_manipulate:
                    # choose manipulation method: PS or DGLM
                    if getattr(args, 'logits_method', None) == 'DGLM' or getattr(args, 'logits_method', None) == 'JSF+PGT+DGLM':
                        # use DGLM per-step logits modulation
                        print("only dglm")
                        from main_logits_attack import generate_dglm
                        final_response_text = generate_dglm(
                            target_lm=targetLM,
                            prompt=orig_msg,
                            d_vocab=d_vocab,
                            max_new_tokens=args.target_max_n_tokens,
                            lambda_=args.lambda_,
                            temperature=args.temperature if hasattr(args, 'temperature') else 0.0,
                            top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
                            dglm_first_k=dglm_first_k,
                        )
                    elif getattr(args, 'logits_method', None) == 'EP+DGLM':
                        # force affirmative prefix first, then apply DGLM
                        from main_logits_attack import generate_ep_dglm
                        final_response_text = generate_ep_dglm(
                            target_lm=targetLM,
                            prompt=orig_msg,
                            prefix=args.target_str,
                            d_vocab=d_vocab,
                            max_new_tokens=args.target_max_n_tokens,
                            lambda_=args.lambda_,
                            temperature=args.temperature if hasattr(args, 'temperature') else 0.0,
                            top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
                        )
                    else:
                        # force affirmative prefix first, then apply DGLM
                        # from main_logits_attack import generate_ep_dglm
                        # final_response_text = generate_ep_dglm(
                        #     target_lm=targetLM,
                        #     prompt=orig_msg,
                        #     prefix=args.target_str,
                        #     d_vocab=d_vocab,
                        #     max_new_tokens=args.target_max_n_tokens,
                        #     lambda_=args.lambda_,
                        #     temperature=args.temperature if hasattr(args, 'temperature') else 0.0,
                        #     top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
                        # )
                        # fallback to PS-style first-token manipulation (or beam_search implementation)
                        from main_logits_attack import generate_ps
                        final_response_text = generate_ps(
                            target_lm=targetLM,
                            prompt=orig_msg,
                            prefix=args.target_str,
                            max_new_tokens=args.target_max_n_tokens,
                            temperature=args.temperature if hasattr(args, 'temperature') else 0.0,
                            top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
                        )
                else:
                    full_prompt = targetLM.get_full_prompt([orig_msg], no_template=False)[0]
                    final_response_text = model_response(full_prompt, targetLM, tokenizer, args.target_max_n_tokens)
                    # final_response_text = targetLM.get_response(prompts_list=[orig_msg], max_n_tokens=args.target_max_n_tokens, temperature=0)[0]['text']
                final_response_texts.append(final_response_text)
                # print("response: ", final_response_text)
                if judge_response(final_response_text) == 1:
                    attack -= 1

                if use_template:
                    writer.writerow([goal, final_response_text])
                else:
                    writer.writerow([init_prompt, final_response_text])
                f.flush()
                os.fsync(f.fileno())

                # print("Current Final response:", final_response_text)

                pbar.set_postfix({'ASR': f'{attack/total:.4f}'})
                pbar.update(1)
        print("Done with all attacks.")
        end_time = time.time()
        print("Total time:", end_time - start_time)
        ###计算攻击成功率
        print(f"Attack success rate: {attack}/{total} = {attack/total:.2%}")

        writer.writerow(['Attack time (s):', (end_time - start_time)/total])
        writer.writerow(['Attack success rate:', f'{attack}/{total} = {attack/total:.2%}'])
        f.flush()
        os.fsync(f.fileno())

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ########### Target model parameters ##########
    parser.add_argument(
        "--target-model",
        default = "gpt-3.5-turbo",
        help = "Name of target model.",
        # choices=["vicuna", "llama2", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview", "claude-instant-1","claude-2", "palm-2"]
    )
    parser.add_argument(
        "--target-max-n-tokens",
        type = int,
        default = 300,
        help = "Maximum number of generated tokens for the target."
    )
    parser.add_argument(
        "--data-path",
        default="dataset/side/refine_prompts_50.csv",
        help="Path to the dataset."
    )
    parser.add_argument(
        "--target-str",
        type=str,
        default="Sure, here is the answer:",
        help="Target string to be fixed in the response."
    )
    parser.add_argument(
        "--use-manipulate",
        type=int,
        default=1,
        help="Whether to use manipulation of the first few tokens."
    )
    parser.add_argument(
        "--use-template",
        type=int,
        default=1,
        help="Whether to use prompt template."
    )
    parser.add_argument(
        "--use-ic",
        type=int,
        default=0,
        help="Whether to use in-context examples."
    )
    parser.add_argument(
        '--logits-method',
        default='DGLM',
        help='When use-manipulate=1, select logits manipulation method (PS, DGLM, or EP+DGLM).'
    )
    parser.add_argument(
        '--direction-dir',
        default='',
        help='Directory with precomputed v_layer_*.npy (used by DGLM).'
    )
    parser.add_argument(
        '--direction-layer',
        default='auto',
        help='Layer index to use for DGLM direction or "auto" to pick last layer.'
    )
    parser.add_argument(
        '--pair-file',
        # default=str(Path(__file__).resolve().parent / 'dataset' / 'side' / 'reverse_pair' / 'harmful_harmless_pair.csv'),
        default=str(PAIR_PATH),
        help='CSV with harmless/harmful pairs to compute direction if no precomputed files.'
    )
    parser.add_argument(
        '--direction-limit',
        type=int,
        default=100,
        help='Number of pairs to use when computing DGLM direction on the fly.'
    )
    parser.add_argument(
        '--lambda',
        dest='lambda_',
        type=float,
        default=50,
        help='DGLM strength lambda.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='sampling temperature used by PS/DGLM when sampling.'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.8,
        help='top-p sampling used by PS/DGLM when sampling.'
    )
    parser.add_argument(
        '--dglm-first-k',
        type=int,
        default=8,
        help='apply DGLM only to the first k generated tokens; 0 means no DGLM.'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        '--save-path',
        default=str(DEFAULT_SAVE_PATH),
        help='Directory to save attack outputs.'
    )
    parser.add_argument(
        '--start',
        '--start-index',
        dest='start_index',
        type=int,
        default=0,
        help='Start from this sample index (0-based). Useful for resuming a run.'
    )
    # parser.add_argument(
    #     "--save-json-path",
    #     type=str,
    #     default="results.json",
    #     help="Path to save the results in JSON format."
    # )

    args = parser.parse_args()
    

    main(args)
