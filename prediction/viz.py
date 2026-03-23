"""
Phân tích bảng NCOD U (ncod_video_id_U_table.json) theo loại câu hỏi.

Lưu ý: U chỉ được học trên train → để có đủ dòng, dùng train.pkl (mặc định).
       Test/valid không có U trong bảng này trừ khi bạn export riêng.

Chạy:
  python viz.py
  python viz.py --split-pkl path/to/valid.pkl
  python viz.py --predictions path/to/preds.json   # join đúng/sai theo qns_key
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# qns_key <-> prediction JSON key (suffix ngắn: _d, _e, _p, _pr, _c, _cr)
# ---------------------------------------------------------------------------
SUFFIX_TO_SHORT: Dict[str, str] = {
    "descriptive": "d",
    "explanatory": "e",
    "predictive": "p",
    "predictive_reason": "pr",
    "counterfactual": "c",
    "counterfactual_reason": "cr",
}

suffix_re = re.compile(
    r"(descriptive|explanatory|predictive_reason|counterfactual_reason|predictive|counterfactual)$"
)


def infer_qtype_and_vid(qns_key: str) -> Tuple[str, str]:
    s = str(qns_key)
    m = suffix_re.search(s)
    if not m:
        return "UNK", s
    qtype_token = m.group(1)
    video_id = s[: m.start(1) - 1]
    mapping = {
        "descriptive": "D",
        "explanatory": "E",
        "predictive": "PA",
        "predictive_reason": "PR",
        "counterfactual": "CA",
        "counterfactual_reason": "CR",
    }
    return mapping[qtype_token], video_id


def qns_key_to_prediction_key(qns_key: str) -> Optional[str]:
    """Chuyển qns_key đầy đủ sang key trong file prediction (vd. *_d, *_pr)."""
    s = str(qns_key)
    m = suffix_re.search(s)
    if not m:
        return None
    token = m.group(1)
    short = SUFFIX_TO_SHORT.get(token)
    if short is None:
        return None
    prefix = s[: m.start(1) - 1]
    return f"{prefix}_{short}"


def lookup_prediction_row(
    qns_key: str, preds: Dict[str, Dict[str, Any]]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Tìm entry trong preds theo thứ tự:
    1) key ngắn: {prefix}_d / _pr / ...
    2) key đầy đủ trùng qns_key (một số notebook export kiểu này)
    """
    pk = qns_key_to_prediction_key(qns_key)
    if pk and pk in preds:
        return pk, preds[pk]
    full = str(qns_key)
    if full in preds:
        return full, preds[full]
    return None, None


def print_prediction_join_diagnostic(
    df: pd.DataFrame, preds: Dict[str, Dict[str, Any]], sample: int = 500
) -> None:
    """Khi 0 khớp: giải thích thường gặp — preds là TEST, U table là TRAIN (khác tập video)."""
    n = min(sample, len(df))
    if n == 0:
        return
    sub = df.head(n)
    hits = 0
    pred_keys_sample: list[str] = []
    for qk in sub["qns_key"]:
        pk = qns_key_to_prediction_key(str(qk))
        pred_keys_sample.append(pk or "")
        if pk and pk in preds:
            hits += 1
    pred_dict_keys = list(preds.keys())

    print("\n[DIAG] Không match predictions — nguyên nhân thường gặp:")
    print("  • File JSON predict thường là **TEST** (hoặc split khác), còn ncod_video_id_U_table.json")
    print("    chỉ chứa **TRAIN** → không có video trùng → 0/N match.")
    print("  • Cách xử lý: export predictions trên **train** (cùng video với lúc học U), hoặc chỉ so U-bin trên val.")
    print(f"  • Trong {n} dòng đầu của U: {hits} pred_key có trong file JSON.")
    if pred_dict_keys:
        print(f"  • Ví dụ key trong preds:  {pred_dict_keys[0]}")
    ex_pk = pred_keys_sample[0] if pred_keys_sample else ""
    print(f"  • Ví dụ pred_key từ U:  {ex_pk or '(không parse được)'}")


def load_u_table(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("data") if isinstance(data, dict) else None
    if rows is None and isinstance(data, list):
        rows = data
    if rows is None:
        raise ValueError("Không tìm được 'data' trong JSON. Kiểm tra schema.")
    df = pd.DataFrame(rows, columns=["sample_idx", "qns_key", "U_value"])
    df["U_value"] = pd.to_numeric(df["U_value"], errors="coerce")
    return df.dropna(subset=["U_value"]).copy()


def load_predictions_json(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_correctness(df: pd.DataFrame, preds: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    pred_keys_used = []
    correct = []
    for qk in out["qns_key"]:
        key_used, row = lookup_prediction_row(str(qk), preds)
        pred_keys_used.append(key_used)
        if row is None:
            correct.append(np.nan)
            continue
        pred = row.get("prediction")
        ans = row.get("answer")
        if pred is None or ans is None:
            correct.append(np.nan)
        else:
            correct.append(1.0 if int(pred) == int(ans) else 0.0)
    out["pred_key"] = pred_keys_used
    out["correct"] = correct
    return out


def u_bin_stats(df: pd.DataFrame, n_bins: int = 4) -> None:
    """In mean accuracy theo quantile của U (chỉ dòng có correct không NaN)."""
    sub = df.dropna(subset=["correct"])
    if sub.empty:
        print("Không có dòng nào join được predictions → bỏ qua U-bin vs accuracy.")
        return
    sub = sub.copy()
    try:
        sub["U_bin"] = pd.qcut(sub["U_value"], q=n_bins, duplicates="drop")
    except ValueError:
        sub["U_bin"] = pd.cut(sub["U_value"], bins=min(n_bins, len(sub)), duplicates="drop")
    g = sub.groupby("U_bin", observed=True)["correct"].agg(["count", "mean"]).reset_index()
    g.columns = ["U_bin", "count", "mean_acc"]
    print("\n=== Mean accuracy theo bin U (quantile) ===")
    print(g.to_string(index=False))


def parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="NCOD U stats by qtype + optional preds merge")
    p.add_argument(
        "--u-json",
        default=os.path.join(here, "ncod_video_id_U_table.json"),
        help="File bảng U (W&B export / notebook)",
    )
    p.add_argument(
        "--split-pkl",
        default=os.path.join(here, "train.pkl"),
        help="PKL chứa list video_id để lọc (NCOD: nên dùng train.pkl)",
    )
    p.add_argument(
        "--split-label",
        default=None,
        help="Nhãn hiển thị (vd. train). Mặc định suy từ tên file: train.pkl -> train",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help=(
            "JSON {key: {prediction, answer}}. Key: *_d/_e/_p/_pr/_c/_cr hoặc full qns_key. "
            "Phải cùng split với bảng U (train preds + train U; test preds không khớp train U)."
        ),
    )
    p.add_argument("--no-plot", action="store_true", help="Chỉ in stats, không show figure")
    p.add_argument("--save-dir", default=None, help="Lưu figure PNG vào thư mục này")
    return p.parse_args()


def split_label_from_path(path: str, override: Optional[str]) -> str:
    if override:
        return override
    base = os.path.basename(path).lower()
    if base.startswith("train"):
        return "train"
    if base.startswith("valid") or base.startswith("val"):
        return "valid"
    if base.startswith("test"):
        return "test"
    return os.path.splitext(base)[0]


def main() -> None:
    args = parse_args()
    split_label = split_label_from_path(args.split_pkl, args.split_label)

    df = load_u_table(args.u_json)
    print("Total U rows:", len(df))
    if len(df):
        print("Example qns_key:", df["qns_key"].iloc[0])

    tmp = df["qns_key"].apply(infer_qtype_and_vid)
    df[["qtype", "video_id"]] = pd.DataFrame(tmp.tolist(), index=df.index)
    print("\nU rows by qtype:")
    print(df["qtype"].value_counts())

    with open(args.split_pkl, "rb") as f:
        split_vids = pickle.load(f)
    split_vids = set(str(x) for x in split_vids)
    print(f"\nNum videos in {split_label}.pkl:", len(split_vids))

    df_sel = df[df["video_id"].isin(split_vids)].copy()
    print(f"U rows with video_id in {split_label} split:", len(df_sel))
    print(df_sel["qtype"].value_counts(dropna=False))

    preds: Optional[Dict[str, Dict[str, Any]]] = None
    if args.predictions:
        preds = load_predictions_json(args.predictions)
        df_sel = merge_correctness(df_sel, preds)
        matched = int(df_sel["correct"].notna().sum())
        print(f"Rows matched to predictions: {matched} / {len(df_sel)}")
        if matched == 0 and len(df_sel) > 0:
            print_prediction_join_diagnostic(df_sel, preds)

    if len(df_sel) == 0:
        print(
            f"\n[WARN] Không có dòng U nào match video_id trong {args.split_pkl}.\n"
            "NCOD U gắn với train samples — dùng train.pkl hoặc kiểm tra định dạng video_id."
        )
        return

    stats = (
        df_sel.groupby("qtype")["U_value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    stats["reason_group"] = stats["qtype"].map(
        {"PR": "Reason", "CR": "Reason", "PA": "Answer", "CA": "Answer", "D": "Other", "E": "Other"}
    )
    print(f"\n=== U stats on {split_label.upper()} split by qtype ===")
    print(stats.sort_values("qtype").to_string(index=False))

    reason_mean = df_sel[df_sel["qtype"].isin(["PR", "CR"])]["U_value"].mean()
    answer_mean = df_sel[df_sel["qtype"].isin(["PA", "CA"])]["U_value"].mean()
    print(f"\nReason mean(U) PR+CR = {reason_mean:.6f}")
    print(f"Answer mean(U) PA+CA = {answer_mean:.6f}")
    if preds is not None:
        u_bin_stats(df_sel)

    order = ["D", "E", "PA", "PR", "CA", "CR"]
    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def _maybe_save(fig: plt.Figure, name: str) -> None:
        if save_dir:
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")

    if not args.no_plot:
        fig1, ax = plt.subplots(figsize=(9, 4.8))
        sns.barplot(data=stats, x="qtype", y="mean", order=order, palette="viridis", ax=ax)
        ax.set_ylabel("Mean U")
        ax.set_xlabel("Question type")
        ax.set_title(f"NCOD mean U by qtype ({split_label})")
        plt.tight_layout()
        _maybe_save(fig1, f"ncod_u_mean_by_qtype_{split_label}.png")
        plt.show()

        fig2, ax = plt.subplots(figsize=(9, 5.2))
        sns.boxplot(data=df_sel, x="qtype", y="U_value", order=order, palette="viridis", ax=ax)
        sns.stripplot(
            data=df_sel,
            x="qtype",
            y="U_value",
            order=order,
            color="black",
            size=1,
            alpha=0.2,
            ax=ax,
        )
        ax.set_ylabel("U_value")
        ax.set_xlabel("Question type")
        ax.set_title(f"NCOD U boxplot ({split_label})")
        plt.tight_layout()
        _maybe_save(fig2, f"ncod_u_box_by_qtype_{split_label}.png")
        plt.show()

        df_sel = df_sel.copy()
        df_sel["group"] = df_sel["qtype"].map(
            {"PR": "PR (Reason)", "CR": "CR (Reason)", "PA": "PA (Answer)", "CA": "CA (Answer)"}
        )
        sub = df_sel[df_sel["qtype"].isin(["PA", "PR", "CA", "CR"])]
        fig3, ax = plt.subplots(figsize=(7, 4.8))
        sns.boxplot(
            data=sub,
            x="group",
            y="U_value",
            order=["PA (Answer)", "PR (Reason)", "CA (Answer)", "CR (Reason)"],
            palette="viridis",
            ax=ax,
        )
        ax.set_ylabel("U_value")
        ax.set_xlabel("Type group")
        ax.set_title(f"Reason vs Answer: U ({split_label})")
        plt.tight_layout()
        _maybe_save(fig3, f"ncod_u_reason_vs_answer_{split_label}.png")
        plt.show()

    print(
        "\nTip: --predictions cần JSON predict **cùng tập video** với bảng U "
        "(vd. train preds khi U là train). File test-predict sẽ cho 0 match."
    )


if __name__ == "__main__":
    main()
