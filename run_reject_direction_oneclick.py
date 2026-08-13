import argparse
import csv
import importlib.util
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PAIR_CSV = PROJECT_ROOT / 'dataset' / 'side' / 'reverse_pair' / 'harmful_harmless_pair.csv'
DEFAULT_QUERY_CSV = PROJECT_ROOT / 'dataset' / 'side' / 'final_prompts_50.csv'

_analysis_path = PROJECT_ROOT / 'analysis' / 'reject_direction_analysis.py'
_spec = importlib.util.spec_from_file_location('reject_direction_analysis', _analysis_path)
_analysis_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_analysis_module)
run_analysis = _analysis_module.run_analysis


def read_safe_unsafe_pair_csv(path: Path, limit: int = 0) -> Tuple[List[str], List[str]]:
    safe_prompts: List[str] = []
    unsafe_prompts: List[str] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            safe = (row.get('harmless') or '').strip()
            unsafe = (row.get('harmful') or '').strip()
            if safe and unsafe:
                safe_prompts.append(safe)
                unsafe_prompts.append(unsafe)
            if limit > 0 and len(safe_prompts) >= limit:
                break
    return safe_prompts, unsafe_prompts


def read_query_csv(path: Path, limit: int = 0) -> List[str]:
    queries: List[str] = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get('final_prompt') or row.get('origin_prompt') or '').strip()
            if q:
                queries.append(q)
            if limit > 0 and len(queries) >= limit:
                break
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description='One-click reject-direction pipeline')
    parser.add_argument('--model', default='llama2-7b', help='target model name')
    parser.add_argument('--pair-csv', default=str(DEFAULT_PAIR_CSV), help='CSV with harmless/harmful columns')
    parser.add_argument('--query-csv', default=str(DEFAULT_QUERY_CSV), help='CSV with final_prompt or origin_prompt')
    parser.add_argument('--pair-limit', type=int, default=50, help='number of safe/unsafe pairs (0 means all)')
    parser.add_argument('--query-limit', type=int, default=50, help='number of query prompts (0 means all)')
    parser.add_argument('--layers', default='', help='comma-separated layers, empty means all')
    parser.add_argument('--out-dir', default='', help='output dir, empty means analysis_output/reject_direction/<model>')
    parser.add_argument('--prefix', default='Sure, here is the answer', help='fixed affirmative prefix to append for prefix-based activations')
    args = parser.parse_args()

    pair_csv = Path(args.pair_csv)
    query_csv = Path(args.query_csv)
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / 'analysis_output' / 'reject_direction' / args.model

    if not pair_csv.exists():
        raise FileNotFoundError(f'pair csv not found: {pair_csv}')
    if not query_csv.exists():
        raise FileNotFoundError(f'query csv not found: {query_csv}')

    safe_prompts, unsafe_prompts = read_safe_unsafe_pair_csv(pair_csv, limit=args.pair_limit)
    queries = read_query_csv(query_csv, limit=args.query_limit)

    if not safe_prompts or not unsafe_prompts:
        raise ValueError('No safe/unsafe prompts loaded from pair-csv.')
    if not queries:
        raise ValueError('No queries loaded from query-csv.')

    layer_indices = [int(x.strip()) for x in args.layers.split(',') if x.strip()] if args.layers else None

    print('One-click reject-direction analysis')
    print(f'  model: {args.model}')
    print(f'  pair-csv: {pair_csv}')
    print(f'  query-csv: {query_csv}')
    print(f'  out-dir: {out_dir}')
    print(f'  safe/unsafe pairs: {len(safe_prompts)}')
    print(f'  queries: {len(queries)}')
    if layer_indices is not None:
        print(f'  layers: {layer_indices}')
    print(f'  prefix: {args.prefix}')

    run_analysis(
        model_name=args.model,
        safe_prompts=safe_prompts,
        unsafe_prompts=unsafe_prompts,
        queries=queries,
        out_dir=str(out_dir),
        layer_indices=layer_indices,
        prefix=args.prefix,
    )


if __name__ == '__main__':
    main()
