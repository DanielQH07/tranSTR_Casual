import argparse
import json
import os
from typing import List, Dict


def load_json(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("entries", [])
    return data


def make_entry(idx: int, text: str, family_bias: List[str], relation_type: str, source: str = "external") -> Dict:
    return {
        "id": f"ck_{idx:06d}",
        "source": source,
        "relation_type": relation_type,
        "question_family_bias": family_bias,
        "keywords": [],
        "text": text,
    }


def convert_legacy_entries(raw_entries, default_family):
    out = []
    for i, e in enumerate(raw_entries):
        if isinstance(e, str):
            out.append(make_entry(i, e, default_family, "transition"))
            continue

        text = e.get("text") or e.get("knowledge") or e.get("template") or ""
        relation = e.get("relation_type", "transition")
        source = e.get("source", "external")
        bias = e.get("question_family_bias", default_family)
        if isinstance(bias, str):
            bias = [bias]
        if not text:
            continue

        out.append(
            {
                "id": e.get("id", f"ck_{i:06d}"),
                "source": source,
                "relation_type": relation,
                "question_family_bias": bias,
                "keywords": e.get("keywords", []),
                "text": text,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Build unified causal knowledge bank")
    parser.add_argument("--pred_bank_path", type=str, default="", help="Legacy predictive bank JSON path")
    parser.add_argument("--cf_bank_path", type=str, default="", help="Legacy counterfactual bank JSON path")
    parser.add_argument("--output_path", type=str, default="data/causal_knowledge_bank.json")
    args = parser.parse_args()

    pred_raw = load_json(args.pred_bank_path) if args.pred_bank_path else []
    cf_raw = load_json(args.cf_bank_path) if args.cf_bank_path else []

    pred_entries = convert_legacy_entries(pred_raw, ["predictive", "predictive_reason"])
    cf_entries = convert_legacy_entries(cf_raw, ["counterfactual", "counterfactual_reason"])

    # Re-index IDs after merge
    merged = []
    for e in pred_entries + cf_entries:
        merged.append(e)

    for i, e in enumerate(merged):
        e["id"] = f"ck_{i:06d}"

    output = {"entries": merged}
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(merged)} entries to {args.output_path}")


if __name__ == "__main__":
    main()
