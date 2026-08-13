import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from tqdm import tqdm  # type: ignore


SIDE_ROOT = Path(__file__).resolve().parent
if str(SIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDE_ROOT))

from conversers import load_target_model  # type: ignore


DEFAULT_TARGET_MODEL = 'llama2-7b'
DEFAULT_PREFIX = 'Sure, here is'
DEFAULT_HARMFUL_FILE = SIDE_ROOT / 'harmful_behaviors' / 'harmful_behaviors_pair.csv'
DEFAULT_SAFE_UNSAFE_FILE = SIDE_ROOT / 'dataset' / 'side' / 'reverse_pair' / 'harmful_harmless_pair.csv'
DEFAULT_PAIR_DATASET_DIR = SIDE_ROOT / 'dataset' / 'Dataset'
DEFAULT_BASELINE_ROOT = SIDE_ROOT / 'dataset' / 'baselines'
DEFAULT_OUT_DIR = SIDE_ROOT / 'analysis_output' / 'harmful_similarity_multi_inputs'
DEFAULT_DIRECTION_DIR = SIDE_ROOT / 'analysis_output' / 'reject_direction'
DEFAULT_ATTACK_SOURCES = ['SIDE', 'GCG', 'AutoDAN', 'Adaptive']
# DEFAULT_ATTACK_SOURCES = ['SIDE']



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_prompts(csv_path: Path, column_names: Sequence[str], limit: Optional[int] = None) -> List[str]:
    prompts: List[str] = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            value = ''
            for col in column_names:
                value = (row.get(col) or '').strip()
                if value:
                    break
            if value:
                prompts.append(value)
            if limit is not None and len(prompts) >= limit:
                break
    return prompts


def read_harmful_prompts(csv_path: Path, limit: Optional[int] = None) -> List[str]:
    return read_csv_prompts(csv_path, ['goal', 'origin_prompt', 'prompt', 'text'], limit=limit)


def read_harmful_harmless_pairs(csv_path: Path, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    harmful: List[str] = []
    harmless: List[str] = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = (row.get('harmful') or '').strip()
            s = (row.get('harmless') or '').strip()
            if h and s:
                harmful.append(h)
                harmless.append(s)
            if limit is not None and len(harmful) >= limit:
                break
    return harmful, harmless


def read_harmful_harmless_pairs_from_dataset_dir(dataset_dir: Path, limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    harmful_csv = dataset_dir / 'Advbench_391_harm.csv'
    harmless_csv = dataset_dir / 'Advbench_391_harmless.csv'

    harmful: List[str] = []
    harmless: List[str] = []
    with open(harmful_csv, 'r', encoding='utf-8', newline='') as f_harm, open(
        harmless_csv, 'r', encoding='utf-8', newline=''
    ) as f_safe:
        harmful_reader = csv.DictReader(f_harm)
        harmless_reader = csv.DictReader(f_safe)
        for harm_row, safe_row in zip(harmful_reader, harmless_reader):
            h = (harm_row.get('text') or '').strip()
            s = (safe_row.get('text') or '').strip()
            if h and s:
                harmful.append(h)
                harmless.append(s)
            if limit is not None and len(harmful) >= limit:
                break
    return harmful, harmless


def read_attack_prompts_from_baseline_folder(folder: Path, model_name: str, limit: Optional[int] = None) -> List[str]:
    csv_path = folder / f'{model_name}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing baseline input file: {csv_path}')
    return read_csv_prompts(csv_path, ['final_prompt', 'prompt', 'text'], limit=limit)


def load_precomputed_directions(direction_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    directions: Dict[int, Dict[str, np.ndarray]] = {}
    v_files = sorted(direction_dir.glob('v_layer_*.npy'), key=lambda p: int(p.stem.split('_')[-1]))
    for path in v_files:
        layer = int(path.stem.split('_')[-1])
        v = np.load(path)
        mu_safe = np.load(direction_dir / f'mu_safe_layer_{layer}.npy')
        mu_unsafe = np.load(direction_dir / f'mu_unsafe_layer_{layer}.npy')
        directions[layer] = {'v_l': v, 'mu_safe': mu_safe, 'mu_unsafe': mu_unsafe}
    return directions


def compute_directions(
    target_lm,
    safe_prompts: List[str],
    unsafe_prompts: List[str],
    layer_indices: Optional[List[int]] = None,
):
    dirs = target_lm.get_reject_directions_per_layer(
        safe_prompts=safe_prompts,
        unsafe_prompts=unsafe_prompts,
        layer_indices=layer_indices,
    )
    return {
        k: {
            'v_l': v['v_l'].detach().cpu().numpy(),
            'mu_safe': v['mu_safe'].detach().cpu().numpy(),
            'mu_unsafe': v['mu_unsafe'].detach().cpu().numpy(),
        }
        for k, v in dirs.items()
    }


@torch.inference_mode()
def compute_similarities_for_prompts(
    target_lm,
    prompts: List[str],
    directions: Dict[int, Dict[str, np.ndarray]],
    prefix: str = '',
):
    tokenizer = target_lm.model.tokenizer
    model = target_lm.model.model
    device = model.device
    model.eval()

    layer_order = sorted(directions.keys())
    sims: List[List[float]] = []

    for prompt in tqdm(prompts, desc='Computing layer similarities'):
        full_prompt = target_lm.get_full_prompt([prompt])[0]
        if prefix:
            full_prompt = full_prompt.rstrip() + ' ' + prefix
        inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=512).to(device)
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask', None),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states

        per_layer: List[float] = []
        for layer in layer_order:
            h = hidden_states[layer][:, -1, :].squeeze(0).cpu()
            h = F.normalize(h, dim=0)
            v = torch.tensor(directions[layer]['v_l'], dtype=h.dtype)
            v = F.normalize(v, dim=0)
            per_layer.append(float(torch.dot(h, v)))
        sims.append(per_layer)

    return np.array(sims), layer_order


def _safe_file_name(text: str, max_len: int = 80) -> str:
    cleaned = ''.join(ch if ch.isalnum() else '_' for ch in text)
    cleaned = cleaned.strip('_')
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned or 'prompt'


def plot_multi_line_mean(
    out_dir: Path,
    layer_order: List[int],
    group_to_sims: Dict[str, np.ndarray],
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    x = np.array(layer_order)
    # 打印x轴的层数
    # print(f'Layer indices on x-axis: {x.tolist()}')
    plt.figure(figsize=(max(8, len(layer_order) * 0.18 + 2.5), 5.2))
    for label, sims in group_to_sims.items():
        mean_vals = sims.mean(axis=0)
        # 打印每个group的层数
        print(f'Group "{label}" - mean similarity layers: {mean_vals.tolist()}')
        plt.plot(x, mean_vals, marker='o', linewidth=2, label=label, markersize=0)
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('layer index', fontsize=22)
    plt.ylabel('cosine similarity', fontsize=22)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'mean_similarity_multi_inputs.png', dpi=150)
    plt.close()


def plot_per_prompt_group(
    out_dir: Path,
    layer_order: List[int],
    group_name: str,
    prompts: List[str],
    sims: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    per_dir = out_dir / 'per_prompt_plots' / group_name
    ensure_dir(per_dir)
    x = np.array(layer_order)

    for i, prompt in enumerate(prompts):
        plt.figure(figsize=(max(7, len(layer_order) * 0.18 + 2), 4.8))
        plt.plot(x, sims[i], marker='o', linewidth=2, label=group_name)
        plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        plt.xlabel('layer index', fontsize=20)
        plt.ylabel('cosine similarity', fontsize=20)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f'{i:03d}_{_safe_file_name(prompt)}.png'
        plt.savefig(per_dir / fname, dpi=150)
        plt.close()


def build_attack_source_map(baseline_root: Path, model_name: str, source_names: List[str]) -> List[Tuple[str, Path]]:
    folder_map = {
        'SIDE': baseline_root / 'baseline_results_side',
        'GCG': baseline_root / 'baseline_results_gcg',
        'AutoDAN': baseline_root / 'baseline_results_autodan',
        'Adaptive': baseline_root / 'baseline_results_adaptive',
    }
    result: List[Tuple[str, Path]] = []
    for name in source_names:
        if name not in folder_map:
            raise ValueError(f'Unknown attack source: {name}. Available: {sorted(folder_map)}')
        result.append((name, folder_map[name] / f'{model_name}.csv'))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute multi-input cosine similarity against harmful directions.')
    parser.add_argument('--target-model', default=DEFAULT_TARGET_MODEL, help='victim model name used by icics_src_side/config.py')
    parser.add_argument('--prefix', default=DEFAULT_PREFIX, help='affirmative prefix appended to SIDE attack inputs before similarity computation')
    parser.add_argument('--harmful-file', default=str(DEFAULT_HARMFUL_FILE), help='CSV containing harmful prompts (goal column)')
    parser.add_argument('--harmless-file', default=str(DEFAULT_SAFE_UNSAFE_FILE), help='CSV containing harmless/harmful pairs (for raw harmless inputs)')
    parser.add_argument(
        '--pair-source',
        choices=['reverse_pair', 'Dataset'],
        default='reverse_pair',
        help='choose paired harmful/harmless data from dataset/side/reverse_pair or dataset/Dataset',
    )
    parser.add_argument('--direction-dir', default='', help='directory with precomputed v_layer_*.npy / mu_safe_layer_*.npy / mu_unsafe_layer_*.npy')
    parser.add_argument('--safe-unsafe-file', default='', help='CSV with harmless/harmful columns; used only when directions must be computed on the fly')
    parser.add_argument('--baseline-root', default=str(DEFAULT_BASELINE_ROOT), help='root folder containing baseline_results_side/gcg/autodan/adaptive')
    parser.add_argument('--attack-sources', default=','.join(DEFAULT_ATTACK_SOURCES), help='comma-separated attack source names: side,gcg,autodan,adaptive')
    parser.add_argument('--out-dir', default='', help='output directory')
    parser.add_argument('--limit', type=int, default=50, help='number of prompts to load per group')
    parser.add_argument('--direction-limit', type=int, default=50, help='number of safe/unsafe pairs used to estimate directions')
    parser.add_argument('--layers', default='', help='comma-separated layer indices to keep; empty means all available layers')
    parser.add_argument('--save-per-prompt-plots', action='store_true', help='save one plot per prompt for each group')
    args = parser.parse_args()

    harmful_file = Path(args.harmful_file)
    harmless_file = Path(args.harmless_file)
    baseline_root = Path(args.baseline_root)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR / args.target_model
    direction_dir = Path(args.direction_dir) if args.direction_dir else Path('')
    safe_unsafe_file = Path(args.safe_unsafe_file) if args.safe_unsafe_file else DEFAULT_SAFE_UNSAFE_FILE

    pair_dataset_dir: Optional[Path]
    if args.pair_source == 'Dataset':
        pair_dataset_dir = DEFAULT_PAIR_DATASET_DIR
        harmless_file = pair_dataset_dir / 'Advbench_391_harmless.csv'
        if not args.safe_unsafe_file:
            safe_unsafe_file = pair_dataset_dir
    else:
        pair_dataset_dir = None

    ensure_dir(out_dir)

    print(f'Loading victim model: {args.target_model}')
    target_lm = load_target_model(SimpleNamespace(target_model=args.target_model))

    harmful_prompts = read_harmful_prompts(harmful_file, limit=args.limit)
    if pair_dataset_dir is not None:
        base_harmful_prompts, harmless_prompts = read_harmful_harmless_pairs_from_dataset_dir(pair_dataset_dir, limit=args.limit)
    else:
        base_harmful_prompts, harmless_prompts = read_harmful_harmless_pairs(harmless_file, limit=args.limit)

    if not harmful_prompts:
        raise ValueError(f'No harmful prompts found in {harmful_file}')
    if not harmless_prompts:
        raise ValueError(f'No harmless prompts found in {harmless_file}')

    attack_source_names = [x.strip() for x in args.attack_sources.split(',') if x.strip()]
    attack_source_files = build_attack_source_map(baseline_root, args.target_model, attack_source_names)

    if args.layers.strip():
        requested_layers = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
    else:
        requested_layers = None

    if direction_dir.exists() and list(direction_dir.glob('v_layer_*.npy')):
        print(f'Loading precomputed harmful directions from: {direction_dir}')
        directions = load_precomputed_directions(direction_dir)
    else:
        print(f'Computing harmful directions from: {safe_unsafe_file}')
        if pair_dataset_dir is not None and safe_unsafe_file == pair_dataset_dir:
            safe_prompts, unsafe_prompts = read_harmful_harmless_pairs_from_dataset_dir(pair_dataset_dir, limit=args.direction_limit)
        else:
            unsafe_prompts, safe_prompts = read_harmful_harmless_pairs(safe_unsafe_file, limit=args.direction_limit)

        try:
            safe_prompts = target_lm.get_full_prompt(safe_prompts)
            unsafe_prompts = target_lm.get_full_prompt(unsafe_prompts)
        except Exception:
            pass

        if not safe_prompts or not unsafe_prompts:
            raise ValueError(f'No safe/unsafe prompts found in {safe_unsafe_file}')
        directions = compute_directions(target_lm, safe_prompts, unsafe_prompts, layer_indices=requested_layers)

    if requested_layers is not None:
        directions = {layer: d for layer, d in directions.items() if layer in requested_layers}
    if not directions:
        raise ValueError('No layer directions available after filtering.')

    group_prompts: Dict[str, List[str]] = {}
    group_prompts['Harmful'] = base_harmful_prompts
    group_prompts['Benign'] = harmless_prompts

    for source_name, csv_path in attack_source_files:
        print(f'Loading attack prompts from: {csv_path}')
        group_prompts[source_name] = read_attack_prompts_from_baseline_folder(csv_path.parent, args.target_model, limit=args.limit)

    group_sims: Dict[str, np.ndarray] = {}
    layer_order: List[int] = []
    # # 打印ddirection的层数
    # print(f'Using directions from layers: {sorted(directions.keys())}')
    for group_name, prompts in group_prompts.items():
        if not prompts:
            raise ValueError(f'No prompts found for group: {group_name}')
        sims, layer_order = compute_similarities_for_prompts(
            target_lm=target_lm,
            prompts=prompts,
            directions=directions,
            prefix=args.prefix if group_name == 'side' else '',
        )
        group_sims[group_name] = sims

        if args.save_per_prompt_plots:
            plot_per_prompt_group(out_dir, layer_order, group_name, prompts, sims)

        np.save(out_dir / f'{group_name}_similarities.npy', sims)

    plot_multi_line_mean(out_dir, layer_order, group_sims)

    summary = {
        'target_model': args.target_model,
        'prefix': args.prefix,
        'pair_source': args.pair_source,
        'harmful_file': str(harmful_file),
        'harmless_file': str(harmless_file),
        'baseline_root': str(baseline_root),
        'attack_sources': attack_source_names,
        'direction_dir': str(direction_dir),
        'layer_order': layer_order,
        'n_groups': len(group_sims),
        'group_sizes': {k: int(v.shape[0]) for k, v in group_sims.items()},
        'group_mean_similarity': {k: v.mean(axis=0).tolist() for k, v in group_sims.items()},
    }
    with open(out_dir / 'multi_inputs_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Done.')
    print(f'Outputs saved to: {out_dir}')
    print('Groups:')
    for name, sims in group_sims.items():
        print(f'  {name}: {sims.shape[0]} prompts')


if __name__ == '__main__':
    main()
