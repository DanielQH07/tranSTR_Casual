"""Evaluation cho NextQA — port từ snippet người dùng cung cấp.

Khác `eval_mc.py` (CausalVidQA) ở chỗ:
- Group theo {CW, CH, TN(+TP), TC, DC, DL, DO}.
- Overall theo family {C, T, D}.
- `qns_id = f"{video}_{qid}"`.
- Result file format JSON: {qid: {"answer": int, "prediction": int}}.
"""

from __future__ import annotations

import json
import os.path as osp
from typing import Dict, Tuple

import pandas as pd

# Tên hiển thị, giữ nguyên format snippet người dùng
MAP_NAME = {
    'CW': 'Why', 'CH': 'How',
    'TN': 'Bef&Aft', 'TC': 'When',
    'DC': 'Cnt', 'DL': 'Loc', 'DO': 'Other',
    'C': 'Acc_C', 'T': 'Acc_T', 'D': 'Acc_D',
}

QTYPES = ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO']


def _load_predictions(result_file_or_dict):
    if isinstance(result_file_or_dict, dict):
        return result_file_or_dict
    with open(result_file_or_dict, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_sample_list(sample_list) -> pd.DataFrame:
    if isinstance(sample_list, pd.DataFrame):
        return sample_list
    return pd.read_csv(sample_list)


def accuracy_metric(sample_list, result_file, verbose: bool = True) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    """Tính accuracy theo NextQA spec.

    Args:
        sample_list: đường dẫn CSV hoặc DataFrame có cột {video, qid, type}.
        result_file: đường dẫn JSON hoặc dict {qid: {answer, prediction}}.
        verbose: có in bảng kết quả không.

    Returns:
        metrics: dict {QTYPE: acc%, FAMILY: acc%, 'Acc': overall%}
        group_acc: dict số correct theo qtype.
        group_cnt: dict số sample theo qtype.
    """
    df = _load_sample_list(sample_list)
    preds = _load_predictions(result_file)

    group: Dict[str, list] = {qt: [] for qt in QTYPES}
    for _, row in df.iterrows():
        qns_id = f"{row['video']}_{row['qid']}"
        qtype = str(row['type'])
        if qtype == 'TP':
            qtype = 'TN'  # gộp như spec
        if qtype in group:
            group[qtype].append(qns_id)

    group_acc: Dict[str, int] = {qt: 0 for qt in QTYPES}
    group_cnt: Dict[str, int] = {qt: 0 for qt in QTYPES}
    overall_acc = {'C': 0, 'T': 0, 'D': 0}
    overall_cnt = {'C': 0, 'T': 0, 'D': 0}
    all_acc = 0
    all_cnt = 0

    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:
            if qid not in preds:
                continue
            cnt += 1
            if preds[qid]['answer'] == preds[qid]['prediction']:
                acc += 1
        group_cnt[qtype] = cnt
        group_acc[qtype] = acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    metrics: Dict[str, float] = {}
    if verbose:
        header = '\t'.join(MAP_NAME[k] for k in QTYPES + ['C', 'T', 'D'])
        print(header)
    line = []
    for qt in QTYPES:
        v = (group_acc[qt] * 100.0 / group_cnt[qt]) if group_cnt[qt] else 0.0
        metrics[qt] = v
        line.append(f'{v:.2f}')
    for fam in ['C', 'T', 'D']:
        v = (overall_acc[fam] * 100.0 / overall_cnt[fam]) if overall_cnt[fam] else 0.0
        metrics[fam] = v
        line.append(f'{v:.2f}')
    if verbose:
        print('\t'.join(line))
    overall = (all_acc * 100.0 / all_cnt) if all_cnt else 0.0
    metrics['Acc'] = overall
    if verbose:
        print(f'Acc: {overall:.2f}')

    return metrics, group_acc, group_cnt


def main(result_file: str, mode: str = 'val', dataset_dir: str = 'dataset/nextqa/') -> None:
    sample_list_file = osp.join(dataset_dir, mode + '.csv')
    print(f'Evaluating {result_file}')
    accuracy_metric(sample_list_file, result_file)


if __name__ == '__main__':
    # ví dụ: python eval_mc_nextqa.py
    model_type = 'HGA'
    mode = 'val'
    model_prefix = f'bert-ft-h256-{mode}-example'
    result_file = f'results/{model_type}-{model_prefix}.json'
    main(result_file, mode)
