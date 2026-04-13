import csv
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "log")

MODELS = ["tranSTR", "Dinov3", "Dinov3_ncod"]


def read_csv_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def pct(a, b):
    if b == 0:
        return 0.0
    return 100.0 * a / b


def confidence_bucket(c):
    if c < 0.4:
        return "<0.4"
    if c < 0.6:
        return "0.4-0.6"
    if c < 0.8:
        return "0.6-0.8"
    if c < 0.9:
        return "0.8-0.9"
    return ">=0.9"


def avg(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_model(model_name):
    incorrect_path = os.path.join(LOG_DIR, model_name, "results_incorrect_all.csv")
    correct_path = os.path.join(LOG_DIR, model_name, "results_correct_all.csv")

    wrong_rows = read_csv_rows(incorrect_path)
    correct_rows = read_csv_rows(correct_path)

    wrong = len(wrong_rows)
    correct = len(correct_rows)
    total = wrong + correct

    wrong_conf = [to_float(r.get("confidence", 0.0)) for r in wrong_rows]
    correct_conf = [to_float(r.get("confidence", 0.0)) for r in correct_rows]

    qtype_wrong_counts = Counter(r.get("question_type", "unknown") for r in wrong_rows)
    qtype_wrong_conf = defaultdict(list)
    for r in wrong_rows:
        qtype_wrong_conf[r.get("question_type", "unknown")].append(to_float(r.get("confidence", 0.0)))

    buckets = Counter(confidence_bucket(c) for c in wrong_conf)

    wrong_ge_09 = sum(1 for c in wrong_conf if c >= 0.9)
    wrong_ge_095 = sum(1 for c in wrong_conf if c >= 0.95)
    correct_ge_09 = sum(1 for c in correct_conf if c >= 0.9)
    correct_ge_095 = sum(1 for c in correct_conf if c >= 0.95)

    return {
        "model": model_name,
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "acc": pct(correct, total),
        "qtype_wrong_counts": qtype_wrong_counts,
        "avg_wrong_conf": avg(wrong_conf),
        "avg_correct_conf": avg(correct_conf),
        "wrong_ge_09": wrong_ge_09,
        "wrong_ge_095": wrong_ge_095,
        "correct_ge_09": correct_ge_09,
        "correct_ge_095": correct_ge_095,
        "buckets": buckets,
        "qtype_wrong_avg_conf": {k: avg(v) for k, v in qtype_wrong_conf.items()},
    }


def summarize_all_models_fail():
    path = os.path.join(LOG_DIR, "videos_all_models_fail.csv")
    rows = read_csv_rows(path)
    n = len(rows)
    fail_rates = [to_float(r.get("fail_rate", 0.0)) for r in rows]
    all_100 = sum(1 for x in fail_rates if x >= 1.0)
    ge_095 = sum(1 for x in fail_rates if x >= 0.95)
    ge_09 = sum(1 for x in fail_rates if x >= 0.9)
    return {
        "rows": n,
        "all_100": all_100,
        "ge_095": ge_095,
        "ge_09": ge_09,
        "avg_fail_rate": avg(fail_rates),
    }


def summarize_hardest_top10():
    path = os.path.join(LOG_DIR, "hardest_videos_top10.csv")
    rows = read_csv_rows(path)
    fail_rates = [to_float(r.get("fail_rate", 0.0)) for r in rows]
    return {
        "rows": len(rows),
        "avg_fail_rate": avg(fail_rates),
        "min_fail_rate": min(fail_rates) if fail_rates else 0.0,
        "max_fail_rate": max(fail_rates) if fail_rates else 0.0,
    }


def summarize_ncod_only_wrong_qtypes():
    path = os.path.join(LOG_DIR, "ncod_only_tr_d3_wrong_by_video.csv")
    rows = read_csv_rows(path)
    qtype_counter = Counter()
    for r in rows:
        qtypes = (r.get("question_types", "") or "").split(",")
        for q in qtypes:
            qq = q.strip()
            if qq:
                qtype_counter[qq] += 1
    return {
        "rows": len(rows),
        "qtype_counter": qtype_counter,
    }


def summarize_comparison_improved_qtypes():
    path = os.path.join(LOG_DIR, "comparison_Dinov3_vs_tranSTR.csv")
    rows = read_csv_rows(path)
    improved = [r for r in rows if (r.get("change", "") or "").strip().lower() == "improved"]
    worsened = [r for r in rows if (r.get("change", "") or "").strip().lower() == "worsened"]

    imp_counter = Counter(r.get("question_type", "unknown") for r in improved)
    wor_counter = Counter(r.get("question_type", "unknown") for r in worsened)

    return {
        "rows": len(rows),
        "improved_rows": len(improved),
        "worsened_rows": len(worsened),
        "improved_qtype_counter": imp_counter,
        "worsened_qtype_counter": wor_counter,
    }


def print_model_report(s):
    print(f"\n=== MODEL: {s['model']} ===")
    print(f"total={s['total']}, correct={s['correct']}, wrong={s['wrong']}, acc={s['acc']:.2f}%")

    print("\n[Wrong by question_type]")
    for q, c in s["qtype_wrong_counts"].most_common():
        print(f"- {q}: {c} ({pct(c, s['wrong']):.2f}% of wrong)")

    print("\n[Confidence summary]")
    print(f"- avg_conf_wrong={s['avg_wrong_conf']:.4f}")
    print(f"- avg_conf_correct={s['avg_correct_conf']:.4f}")
    print(
        f"- wrong_conf>=0.9: {s['wrong_ge_09']}/{s['wrong']} ({pct(s['wrong_ge_09'], s['wrong']):.2f}%)"
    )
    print(
        f"- wrong_conf>=0.95: {s['wrong_ge_095']}/{s['wrong']} ({pct(s['wrong_ge_095'], s['wrong']):.2f}%)"
    )
    print(
        f"- correct_conf>=0.9: {s['correct_ge_09']}/{s['correct']} ({pct(s['correct_ge_09'], s['correct']):.2f}%)"
    )
    print(
        f"- correct_conf>=0.95: {s['correct_ge_095']}/{s['correct']} ({pct(s['correct_ge_095'], s['correct']):.2f}%)"
    )

    print("\n[Wrong confidence buckets]")
    ordered = ["<0.4", "0.4-0.6", "0.6-0.8", "0.8-0.9", ">=0.9"]
    for b in ordered:
        c = s["buckets"].get(b, 0)
        print(f"- {b}: {c} ({pct(c, s['wrong']):.2f}%)")

    print("\n[Per-question_type avg wrong confidence]")
    sorted_items = sorted(
        s["qtype_wrong_avg_conf"].items(),
        key=lambda kv: (-s["qtype_wrong_counts"].get(kv[0], 0), -kv[1], kv[0]),
    )
    for q, conf in sorted_items:
        print(f"- {q}: avg_wrong_conf={conf:.4f}, wrong_count={s['qtype_wrong_counts'][q]}")

    # Upper-bound style estimate requested by user
    wrong_low_conf = s["wrong"] - s["wrong_ge_09"]
    print("\n[Verifier utility estimate]")
    print(
        "- If verifier/recheck only triggers when conf < 0.9: "
        f"max addressable wrong cases = {wrong_low_conf}/{s['wrong']} ({pct(wrong_low_conf, s['wrong']):.2f}%)"
    )
    print(
        "- High-conf wrong (conf >= 0.9) likely bypasses simple confidence gate: "
        f"{s['wrong_ge_09']}/{s['wrong']} ({pct(s['wrong_ge_09'], s['wrong']):.2f}%)"
    )


def print_aux_reports():
    fail = summarize_all_models_fail()
    hard = summarize_hardest_top10()
    ncod = summarize_ncod_only_wrong_qtypes()
    cmp_ = summarize_comparison_improved_qtypes()

    print("\n=== HARD VIDEOS / CROSS-MODEL DIFFICULTY ===")
    print(f"videos_all_models_fail rows={fail['rows']}")
    print(f"- fail_rate==1.0: {fail['all_100']}")
    print(f"- fail_rate>=0.95: {fail['ge_095']}")
    print(f"- fail_rate>=0.90: {fail['ge_09']}")
    print(f"- avg_fail_rate: {fail['avg_fail_rate']:.4f}")

    print("\n[hardest_videos_top10 summary]")
    print(f"- rows: {hard['rows']}")
    print(f"- avg_fail_rate: {hard['avg_fail_rate']:.4f}")
    print(f"- min_fail_rate: {hard['min_fail_rate']:.4f}")
    print(f"- max_fail_rate: {hard['max_fail_rate']:.4f}")

    print("\n[ncod_only_tr_d3_wrong_by_video: question_type frequency]")
    print(f"- rows: {ncod['rows']}")
    for q, c in ncod["qtype_counter"].most_common():
        print(f"- {q}: {c}")

    print("\n[comparison_Dinov3_vs_tranSTR]")
    print(f"- total rows: {cmp_['rows']}")
    print(f"- improved rows: {cmp_['improved_rows']}")
    print(f"- worsened rows: {cmp_['worsened_rows']}")

    print("\n- improved by question_type")
    for q, c in cmp_["improved_qtype_counter"].most_common():
        print(f"  - {q}: {c}")

    if cmp_["worsened_rows"] > 0:
        print("\n- worsened by question_type")
        for q, c in cmp_["worsened_qtype_counter"].most_common():
            print(f"  - {q}: {c}")


def main():
    print("Schema check from CSV headers has been done separately.")
    for m in MODELS:
        s = summarize_model(m)
        print_model_report(s)

    print_aux_reports()

    print("\n=== QUICK TAKEAWAY TEMPLATE ===")
    print("1) Question types causing most failures are those with highest wrong_count in each model.")
    print("2) If wrong_conf>=0.9 is high, model has overconfident errors and plain verifier gating by confidence is limited.")
    print("3) If wrong_conf mostly in 0.4-0.8, verifier/rerank can help substantially.")


if __name__ == "__main__":
    main()
