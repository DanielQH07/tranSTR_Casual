"""
Generate evaluation.ipynb in the log/ directory.
Run once:  python create_evaluation.py
"""
import json, os

def code_cell(source, md=False):
    return {
        "cell_type": "markdown" if md else "code",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        **({"execution_count": None, "outputs": []} if not md else {}),
    }

cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN: Title
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell(
    "# 📊 TranSTR Model Evaluation & Comparison\n"
    "\n"
    "This notebook:\n"
    "1. **Visualizes distribution** of each model's results (accuracy, confidence, per question type)\n"
    "2. **Compares models** — what they get right/wrong in common, and how they differ\n"
    "3. **Breaks down** analysis by question type (D, E, P, PR, C, CR)",
    md=True
))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Imports & Config
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Config ──
LOG_DIR = Path(".")  # log/ directory
QTYPES = ["descriptive", "explanatory", "predictive", "predictive_reason",
          "counterfactual", "counterfactual_reason"]
QTYPE_SHORT = {"descriptive": "D", "explanatory": "E", "predictive": "P",
               "predictive_reason": "PR", "counterfactual": "C",
               "counterfactual_reason": "CR"}

# ── Style ──
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'figure.dpi': 120,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

MODEL_COLORS = {
    'tranSTR': '#58a6ff',
    'Dinov2': '#f778ba',
    'Dinov3': '#7ee787',
    'CLIP': '#d2a8ff',
    'SigCLIP': '#ffa657',
    'TokenMark': '#ff7b72',
}

print("✅ Config loaded")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Load Data
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Auto-discover models with CSV files ──
model_data = {}
for model_dir in sorted(LOG_DIR.iterdir()):
    if not model_dir.is_dir():
        continue
    correct_csv = model_dir / "results_correct_all.csv"
    incorrect_csv = model_dir / "results_incorrect_all.csv"
    if correct_csv.exists() and incorrect_csv.exists():
        df_c = pd.read_csv(correct_csv)
        df_i = pd.read_csv(incorrect_csv)
        df = pd.concat([df_c, df_i], ignore_index=True)
        # ── Force correct types ──
        df['is_correct'] = df['is_correct'].astype(str).str.strip().str.lower() == 'true'
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
        df['model'] = model_dir.name

        # ── Validation: duplicate keys may cause merge issues ──
        key_cols = ['video_id', 'question_type', 'question']
        if all(col in df.columns for col in key_cols):
            dup_count = int(df.duplicated(subset=key_cols, keep=False).sum())
            if dup_count > 0:
                print(f"  ⚠️ {model_dir.name}: {dup_count} duplicate rows for KEY_COLS {key_cols} (will be dropped during merge).")

        # ── Fallback: recompute confidence from per-answer probabilities if flat ──
        prob_cols = [f'prob_a{i}' for i in range(5)]
        if all(col in df.columns for col in prob_cols):
            conf_unique = df['confidence'].dropna().unique()
            if len(conf_unique) == 1:
                try:
                    probs = df[prob_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                    df['confidence'] = probs.max(axis=1)
                    print(f"  ℹ️ {model_dir.name}: confidence was flat; recomputed from {prob_cols}.")
                except Exception as e:
                    print(f"  ⚠️ {model_dir.name}: failed to recompute confidence from probabilities: {e}")

        model_data[model_dir.name] = df
        print(f"  📁 {model_dir.name}: {len(df_c)} correct + {len(df_i)} incorrect = {len(df)} total")

MODEL_NAMES = list(model_data.keys())
df_all = pd.concat(model_data.values(), ignore_index=True)
print(f"\n✅ Loaded {len(MODEL_NAMES)} models: {MODEL_NAMES}")
print(f"   Total records: {len(df_all):,}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN: Section 1
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell(
    "---\n"
    "## 📈 Part 1: Distribution Visualization per Model",
    md=True
))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Overall Accuracy Summary
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Overall Accuracy Table ──
summary_rows = []
for name in MODEL_NAMES:
    df = model_data[name]
    total = len(df)
    correct = df['is_correct'].sum()
    acc = correct / total * 100
    avg_conf = df['confidence'].mean()
    avg_conf_correct = df[df['is_correct']]['confidence'].mean()
    avg_conf_incorrect = df[~df['is_correct']]['confidence'].mean()
    summary_rows.append({
        'Model': name, 'Total': total, 'Correct': int(correct),
        'Incorrect': total - int(correct),
        'Accuracy (%)': round(acc, 2),
        'Avg Confidence': round(avg_conf, 4),
        'Avg Conf (Correct)': round(avg_conf_correct, 4),
        'Avg Conf (Incorrect)': round(avg_conf_incorrect, 4),
    })

df_summary = pd.DataFrame(summary_rows).sort_values('Accuracy (%)', ascending=False)
print("=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)
print(df_summary.to_string(index=False))
print()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Overall Accuracy Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Overall Accuracy Bar Chart ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
colors = [MODEL_COLORS.get(m, '#58a6ff') for m in df_summary['Model']]
bars = axes[0].barh(df_summary['Model'], df_summary['Accuracy (%)'], color=colors, edgecolor='white', linewidth=0.5, height=0.6)
for bar, acc in zip(bars, df_summary['Accuracy (%)']):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{acc:.2f}%',
                va='center', fontweight='bold', fontsize=12, color='#c9d1d9')
axes[0].set_xlabel('Accuracy (%)')
axes[0].set_title('Overall Accuracy', fontsize=14, fontweight='bold', color='white')
axes[0].set_xlim(0, max(df_summary['Accuracy (%)']) + 5)

# Confidence comparison
x_pos = np.arange(len(df_summary))
w = 0.35
bars1 = axes[1].bar(x_pos - w/2, df_summary['Avg Conf (Correct)'], w, label='Correct', color='#7ee787', alpha=0.85)
bars2 = axes[1].bar(x_pos + w/2, df_summary['Avg Conf (Incorrect)'], w, label='Incorrect', color='#ff7b72', alpha=0.85)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(df_summary['Model'], rotation=30, ha='right')
axes[1].set_ylabel('Average Confidence')
axes[1].set_title('Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold', color='white')
axes[1].legend(facecolor='#161b22', edgecolor='#30363d')

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Accuracy by Question Type (Grouped Bar)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Accuracy per Question Type ──
qtype_acc = {}
for name in MODEL_NAMES:
    df = model_data[name]
    accs = {}
    for qt in QTYPES:
        sub = df[df['question_type'] == qt]
        if len(sub) > 0:
            accs[QTYPE_SHORT[qt]] = sub['is_correct'].mean() * 100
        else:
            accs[QTYPE_SHORT[qt]] = 0
    qtype_acc[name] = accs

df_qtype = pd.DataFrame(qtype_acc).T
print("Accuracy by Question Type (%):")
print(df_qtype.round(2).to_string())
print()

# ── Grouped bar chart ──
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(QTYPE_SHORT))
n_models = len(MODEL_NAMES)
total_width = 0.75
bar_width = total_width / n_models

for i, name in enumerate(MODEL_NAMES):
    offset = (i - n_models/2 + 0.5) * bar_width
    color = MODEL_COLORS.get(name, '#58a6ff')
    vals = [qtype_acc[name][s] for s in QTYPE_SHORT.values()]
    bars = ax.bar(x + offset, vals, bar_width, label=name, color=color, edgecolor='white', linewidth=0.3, alpha=0.9)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7, color=color, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f"{s}\\n({f})" for f, s in QTYPE_SHORT.items()], fontsize=9)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy by Question Type per Model', fontsize=15, fontweight='bold', color='white')
ax.legend(facecolor='#161b22', edgecolor='#30363d', fontsize=9)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Heatmap: Model x Question Type ──
fig, ax = plt.subplots(figsize=(10, max(4, len(MODEL_NAMES) * 1.2)))
sns.heatmap(df_qtype, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5,
            ax=ax, cbar_kws={'label': 'Accuracy (%)'}, vmin=0, vmax=100)
ax.set_title('Accuracy Heatmap: Model × Question Type', fontsize=14, fontweight='bold', color='white')
ax.set_ylabel('Model')
ax.set_xlabel('Question Type')
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Confidence Distribution
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Confidence Distribution per Model (Correct vs Incorrect) ──
n = len(MODEL_NAMES)
fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
if n == 1:
    axes = [axes]

for i, name in enumerate(MODEL_NAMES):
    ax = axes[i]
    df = model_data[name]
    ax.hist(df[df['is_correct']]['confidence'], bins=50, alpha=0.7, color='#7ee787', label='Correct', density=True)
    ax.hist(df[~df['is_correct']]['confidence'], bins=50, alpha=0.7, color='#ff7b72', label='Incorrect', density=True)
    ax.set_title(name, fontsize=13, fontweight='bold', color=MODEL_COLORS.get(name, 'white'))
    ax.set_xlabel('Confidence')
    if i == 0:
        ax.set_ylabel('Density')
    ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')

fig.suptitle('Confidence Distribution: Correct vs Incorrect', fontsize=15, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: Per-question-type distribution
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Sample count per question type per model ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Stacked bar: question type distribution
qtype_counts = {}
for name in MODEL_NAMES:
    df = model_data[name]
    counts = df['question_type'].value_counts()
    qtype_counts[name] = {QTYPE_SHORT.get(qt, qt): counts.get(qt, 0) for qt in QTYPES}

df_counts = pd.DataFrame(qtype_counts).T
df_counts.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2', edgecolor='white', linewidth=0.3)
axes[0].set_title('Samples per Question Type', fontsize=13, fontweight='bold', color='white')
axes[0].set_ylabel('Count')
axes[0].legend(facecolor='#161b22', edgecolor='#30363d', fontsize=8)
axes[0].tick_params(axis='x', rotation=30)

# Correct rate by question type (line plot)
for name in MODEL_NAMES:
    vals = [qtype_acc[name][s] for s in QTYPE_SHORT.values()]
    axes[1].plot(list(QTYPE_SHORT.values()), vals, 'o-', label=name,
                color=MODEL_COLORS.get(name, '#58a6ff'), linewidth=2, markersize=8)

axes[1].set_title('Accuracy Trend by Question Type', fontsize=13, fontweight='bold', color='white')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(facecolor='#161b22', edgecolor='#30363d', fontsize=9)
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN: Section 2
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell(
    "---\n"
    "## 🔍 Part 2: Cross-Model Comparison\n"
    "\n"
    "For each question (identified by `video_id` + `question_type` + `question`), "
    "compare whether models agree or disagree on the answer.",
    md=True
))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Build comparison DataFrame
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Build wide-format comparison table ──
# Each row = one question, columns = is_correct for each model
KEY_COLS = ['video_id', 'question_type', 'question']

merged = None
for name in MODEL_NAMES:
    df = model_data[name][KEY_COLS + ['is_correct', 'predicted_idx', 'confidence']].copy()
    df = df.drop_duplicates(subset=KEY_COLS)  # prevent many-to-many merge
    df = df.rename(columns={
        'is_correct': f'correct_{name}',
        'predicted_idx': f'pred_{name}',
        'confidence': f'conf_{name}',
    })
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=KEY_COLS, how='outer')

print(f"Total unique questions: {len(merged):,}")
print(f"Models: {MODEL_NAMES}")

# ── Classify each question ──
correct_cols = [f'correct_{m}' for m in MODEL_NAMES]

merged['all_correct'] = merged[correct_cols].all(axis=1)
merged['all_incorrect'] = (~merged[correct_cols]).all(axis=1)
merged['some_disagree'] = ~merged['all_correct'] & ~merged['all_incorrect']

n_all_c = merged['all_correct'].sum()
n_all_i = merged['all_incorrect'].sum()
n_disagree = merged['some_disagree'].sum()

print(f"\\n{'='*60}")
print(f"  ✅ ALL models correct:    {n_all_c:>6,} ({n_all_c/len(merged)*100:.1f}%)")
print(f"  ❌ ALL models incorrect:  {n_all_i:>6,} ({n_all_i/len(merged)*100:.1f}%)")
print(f"  ⚡ Models DISAGREE:       {n_disagree:>6,} ({n_disagree/len(merged)*100:.1f}%)")
print(f"{'='*60}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: Agreement Pie + Bar
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Agreement Overview ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
sizes = [n_all_c, n_all_i, n_disagree]
labels = [f'All Correct\\n{n_all_c:,}', f'All Incorrect\\n{n_all_i:,}', f'Disagree\\n{n_disagree:,}']
colors_pie = ['#7ee787', '#ff7b72', '#ffa657']
explode = (0.03, 0.03, 0.06)
wedges, texts, autotexts = axes[0].pie(sizes, labels=labels, colors=colors_pie, explode=explode,
    autopct='%1.1f%%', startangle=90, textprops={'color': '#c9d1d9', 'fontsize': 11})
for at in autotexts:
    at.set_fontweight('bold')
axes[0].set_title('Cross-Model Agreement', fontsize=14, fontweight='bold', color='white')

# Per question type breakdown
cats = ['all_correct', 'all_incorrect', 'some_disagree']
cat_labels = ['All Correct', 'All Incorrect', 'Disagree']
cat_colors = ['#7ee787', '#ff7b72', '#ffa657']

qtype_breakdown = {}
for qt in QTYPES:
    sub = merged[merged['question_type'] == qt]
    qtype_breakdown[QTYPE_SHORT[qt]] = [sub[c].sum() for c in cats]

df_breakdown = pd.DataFrame(qtype_breakdown, index=cat_labels).T
df_breakdown.plot(kind='bar', stacked=True, ax=axes[1], color=cat_colors, edgecolor='white', linewidth=0.3)
axes[1].set_title('Agreement by Question Type', fontsize=14, fontweight='bold', color='white')
axes[1].set_ylabel('Count')
axes[1].legend(facecolor='#161b22', edgecolor='#30363d')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11: Pairwise Agreement Matrix
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Pairwise Agreement Matrix ──
n_models = len(MODEL_NAMES)
agree_matrix = np.zeros((n_models, n_models))
disagree_detail = {}

for i, m1 in enumerate(MODEL_NAMES):
    for j, m2 in enumerate(MODEL_NAMES):
        col1 = f'correct_{m1}'
        col2 = f'correct_{m2}'
        valid = merged[[col1, col2]].dropna()
        agree = float((valid[col1].values == valid[col2].values).mean() * 100)
        agree_matrix[i, j] = agree

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.zeros_like(agree_matrix, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(agree_matrix, xticklabels=MODEL_NAMES, yticklabels=MODEL_NAMES,
            annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, mask=mask,
            vmin=50, vmax=100, linewidths=0.5,
            cbar_kws={'label': 'Agreement (%)'})
# Fill diagonal with 100
for i in range(n_models):
    ax.text(i + 0.5, i + 0.5, '100.0', ha='center', va='center',
            fontweight='bold', fontsize=11, color='#c9d1d9')
ax.set_title('Pairwise Agreement Matrix (%)', fontsize=14, fontweight='bold', color='white')
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: Pairwise Detailed Comparison
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Pairwise: Who is better on disagreed questions? ──
print("=" * 80)
print("PAIRWISE COMPARISON: On questions where models DISAGREE")
print("=" * 80)

pair_rows = []
for m1, m2 in combinations(MODEL_NAMES, 2):
    c1 = f'correct_{m1}'
    c2 = f'correct_{m2}'
    valid = merged[[c1, c2, 'question_type']].dropna()
    disagree = valid[valid[c1] != valid[c2]]
    m1_wins = (disagree[c1] & ~disagree[c2]).sum()
    m2_wins = (~disagree[c1] & disagree[c2]).sum()
    total_disagree = len(disagree)
    
    print(f"\\n  {m1} vs {m2}:")
    print(f"    Total disagreements: {total_disagree}")
    print(f"    {m1} correct, {m2} wrong: {m1_wins} ({m1_wins/max(total_disagree,1)*100:.1f}%)")
    print(f"    {m2} correct, {m1} wrong: {m2_wins} ({m2_wins/max(total_disagree,1)*100:.1f}%)")
    
    # Per question type
    for qt in QTYPES:
        sub = disagree[disagree['question_type'] == qt]
        w1 = (sub[c1] & ~sub[c2]).sum()
        w2 = (~sub[c1] & sub[c2]).sum()
        pair_rows.append({'Pair': f'{m1} vs {m2}', 'QType': QTYPE_SHORT[qt],
                         f'{m1}_wins': w1, f'{m2}_wins': w2, 'total': len(sub)})

print()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13: Disagreement visualization per pair
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Visualize pairwise disagreements by question type ──
pairs = list(combinations(MODEL_NAMES, 2))
n_pairs = len(pairs)
fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 6), squeeze=False)

for idx, (m1, m2) in enumerate(pairs):
    ax = axes[0][idx]
    c1 = f'correct_{m1}'
    c2 = f'correct_{m2}'
    valid = merged[[c1, c2, 'question_type']].dropna()
    disagree = valid[valid[c1] != valid[c2]]
    
    m1_wins_per_qt = []
    m2_wins_per_qt = []
    for qt in QTYPES:
        sub = disagree[disagree['question_type'] == qt]
        m1_wins_per_qt.append((sub[c1] & ~sub[c2]).sum())
        m2_wins_per_qt.append((~sub[c1] & sub[c2]).sum())
    
    x = np.arange(len(QTYPES))
    w = 0.35
    ax.bar(x - w/2, m1_wins_per_qt, w, label=f'{m1} wins', color=MODEL_COLORS.get(m1, '#58a6ff'), alpha=0.85)
    ax.bar(x + w/2, m2_wins_per_qt, w, label=f'{m2} wins', color=MODEL_COLORS.get(m2, '#f778ba'), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([QTYPE_SHORT[qt] for qt in QTYPES], fontsize=9)
    ax.set_title(f'{m1} vs {m2}', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('# Questions')
    ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')

fig.suptitle('Pairwise Disagreements by Question Type', fontsize=15, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN: Section 3
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell(
    "---\n"
    "## 🎯 Part 3: Deep Dive — Per Question Type Analysis",
    md=True
))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: Per question type deep dive
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Per Question Type: Agreement stats ──
for qt in QTYPES:
    sub = merged[merged['question_type'] == qt]
    total = len(sub)
    ac = sub['all_correct'].sum()
    ai = sub['all_incorrect'].sum()
    dis = sub['some_disagree'].sum()
    
    print(f"\\n{'='*60}")
    print(f"  Question Type: {qt} ({QTYPE_SHORT[qt]})")
    print(f"  Total questions: {total}")
    print(f"  ✅ All correct:   {ac:>5} ({ac/total*100:.1f}%)")
    print(f"  ❌ All incorrect: {ai:>5} ({ai/total*100:.1f}%)")
    print(f"  ⚡ Disagree:      {dis:>5} ({dis/total*100:.1f}%)")
    
    # Per model accuracy on this question type
    for name in MODEL_NAMES:
        col = f'correct_{name}'
        valid = sub[col].dropna()
        acc = valid.mean() * 100
        print(f"    {name:>12}: {acc:.2f}%")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15: Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Radar Chart: Model Performance Profile ──
from math import pi

categories = list(QTYPE_SHORT.values())
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_facecolor('#161b22')

for name in MODEL_NAMES:
    values = [qtype_acc[name][s] for s in categories]
    values += values[:1]
    color = MODEL_COLORS.get(name, '#58a6ff')
    ax.plot(angles, values, 'o-', linewidth=2, label=name, color=color)
    ax.fill(angles, values, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, color='#c9d1d9')
ax.set_ylim(0, 100)
ax.set_title('Model Performance Profile (Radar)', fontsize=15, fontweight='bold',
             color='white', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#161b22', edgecolor='#30363d')
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16: Unique strengths
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Unique Strengths: Questions ONLY this model gets right ──
print("=" * 80)
print("UNIQUE STRENGTHS: Questions only ONE model answers correctly")
print("=" * 80)

for name in MODEL_NAMES:
    other_models = [m for m in MODEL_NAMES if m != name]
    this_correct = merged[f'correct_{name}'] == True
    others_all_wrong = True
    for om in other_models:
        others_all_wrong = others_all_wrong & (merged[f'correct_{om}'] == False)
    
    unique = merged[this_correct & others_all_wrong]
    print(f"\\n  🌟 {name}: {len(unique)} unique correct answers")
    
    # Breakdown by question type
    for qt in QTYPES:
        count = len(unique[unique['question_type'] == qt])
        if count > 0:
            print(f"      {QTYPE_SHORT[qt]}: {count}")

print()
print("=" * 80)
print("UNIQUE WEAKNESSES: Questions only ONE model gets WRONG")
print("=" * 80)

for name in MODEL_NAMES:
    other_models = [m for m in MODEL_NAMES if m != name]
    this_wrong = merged[f'correct_{name}'] == False
    others_all_correct = True
    for om in other_models:
        others_all_correct = others_all_correct & (merged[f'correct_{om}'] == True)
    
    unique_weak = merged[this_wrong & others_all_correct]
    print(f"\\n  💔 {name}: {len(unique_weak)} unique incorrect answers")
    for qt in QTYPES:
        count = len(unique_weak[unique_weak['question_type'] == qt])
        if count > 0:
            print(f"      {QTYPE_SHORT[qt]}: {count}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 17: Unique strengths visualization
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Visualize Unique Strengths/Weaknesses ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

strengths = {}
weaknesses = {}
for name in MODEL_NAMES:
    other_models = [m for m in MODEL_NAMES if m != name]
    this_c = merged[f'correct_{name}'] == True
    this_w = merged[f'correct_{name}'] == False
    ow = True
    oc = True
    for om in other_models:
        ow = ow & (merged[f'correct_{om}'] == False)
        oc = oc & (merged[f'correct_{om}'] == True)
    strengths[name] = (this_c & ow).sum()
    weaknesses[name] = (this_w & oc).sum()

colors = [MODEL_COLORS.get(m, '#58a6ff') for m in MODEL_NAMES]
axes[0].bar(MODEL_NAMES, [strengths[m] for m in MODEL_NAMES], color=colors, edgecolor='white', linewidth=0.5)
for i, (m, v) in enumerate(strengths.items()):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold', color='#c9d1d9')
axes[0].set_title('🌟 Unique Strengths\\n(Only this model correct)', fontsize=13, fontweight='bold', color='white')
axes[0].set_ylabel('# Questions')
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(MODEL_NAMES, [weaknesses[m] for m in MODEL_NAMES], color=colors, edgecolor='white', linewidth=0.5)
for i, (m, v) in enumerate(weaknesses.items()):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold', color='#c9d1d9')
axes[1].set_title('💔 Unique Weaknesses\\n(Only this model wrong)', fontsize=13, fontweight='bold', color='white')
axes[1].set_ylabel('# Questions')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 18: Export summary
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Export comparison summary to CSV ──
# Save the merged comparison table
merged.to_csv('comparison_all_models.csv', index=False)
print(f"✅ Saved comparison_all_models.csv ({len(merged)} rows)")

# Save per question type accuracy
df_qtype.to_csv('accuracy_per_qtype.csv')
print(f"✅ Saved accuracy_per_qtype.csv")

# Save summary
df_summary.to_csv('model_summary.csv', index=False)
print(f"✅ Saved model_summary.csv")

print("\\n🎉 Evaluation complete!")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN: Section 4
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell(
    "---\n"
    "## 🔬 Part 4: Frame & Object Rationalization Analysis\n"
    "\n"
    "Compare **which frames** each model selects (temporal rationalization) "
    "and **which objects** within those frames receive attention (spatial rationalization).\n"
    "\n"
    "> Requires `frame_scores`, `selected_frame_indices`, and `obj_scores_per_frame` columns in CSVs.",
    md=True
))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 19: Check availability & show sample video_ids
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Check if frame/object data is available ──
import json as _json

FRAME_COLS = ['frame_scores', 'selected_frame_indices', 'obj_scores_per_frame']
models_with_frames = {}

for name in MODEL_NAMES:
    df = model_data[name]
    has_cols = all(c in df.columns for c in FRAME_COLS)
    has_data = False
    if has_cols:
        has_data = df['frame_scores'].notna().any()
    models_with_frames[name] = has_cols and has_data
    status = '✅' if (has_cols and has_data) else '❌'
    print(f"  {status} {name}: frame/object data {'available' if has_cols and has_data else 'NOT available'}")

FRAME_MODELS = [m for m, v in models_with_frames.items() if v]
print(f"\\nModels with frame data: {FRAME_MODELS if FRAME_MODELS else 'NONE'}")

if not FRAME_MODELS:
    print("\\n⚠️  No model has frame/object data yet.")
    print("    Re-run inference with updated add_inf.py to generate these columns.")
    print("    The remaining cells in this section will be skipped.")
else:
    # Show 10 sample video_ids that all frame-models share
    common_vids = set(model_data[FRAME_MODELS[0]]['video_id'].unique())
    for m in FRAME_MODELS[1:]:
        common_vids &= set(model_data[m]['video_id'].unique())
    sample_vids = sorted(common_vids)[:10]
    print(f"\\n📋 Sample video_ids (shared across all frame-models):")
    for i, vid in enumerate(sample_vids):
        print(f"   {i+1}. {vid}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 20: User config — pick video_id & video directory
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ══════════════════════════════════════════════════════════════
# 👇 USER: Set your video_id and path to the video folder here
# ══════════════════════════════════════════════════════════════
TARGET_VIDEO_ID = "CHANGE_ME"   # e.g. "I3lGH_9Rieg_000001_000011"
VIDEO_DIR = r"CHANGE_ME"        # e.g. r"D:\\data\\videos"  (contains {video_id}.mp4)
NUM_FRAMES = 16                 # number of sampled frames

print(f"Target video: {TARGET_VIDEO_ID}")
print(f"Video dir:    {VIDEO_DIR}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 21: Extract & display frames from video
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Extract frames from original video ──
import cv2
from pathlib import Path as _Path

video_path = _Path(VIDEO_DIR) / f"{TARGET_VIDEO_ID}.mp4"
if not video_path.exists():
    # try .avi / .webm
    for ext in ['.avi', '.webm', '.mkv']:
        alt = video_path.with_suffix(ext)
        if alt.exists():
            video_path = alt
            break

assert video_path.exists(), f"Video not found: {video_path}"

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {video_path.name}  |  {total_frames} frames  |  {fps:.1f} fps  |  {total_frames/fps:.1f}s")

# Sample NUM_FRAMES uniformly
frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
all_frames = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        all_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
cap.release()

# Display all sampled frames
fig, axes = plt.subplots(2, NUM_FRAMES // 2, figsize=(NUM_FRAMES * 2, 6))
for i, (ax, frame) in enumerate(zip(axes.flat, all_frames)):
    ax.imshow(frame)
    ax.set_title(f'F{i}', fontsize=9, color='#c9d1d9')
    ax.axis('off')
fig.suptitle(f'All {NUM_FRAMES} Sampled Frames — {TARGET_VIDEO_ID}',
             fontsize=14, fontweight='bold', color='white')
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 22: Compare frame selections across models
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Compare frame selections across models for TARGET_VIDEO_ID ──
if not FRAME_MODELS:
    print("⚠️ No frame data available. Skipping.")
else:
    print(f"\\n{'='*80}")
    print(f"FRAME SELECTION COMPARISON — {TARGET_VIDEO_ID}")
    print(f"{'='*80}")
    
    model_frame_data = {}  # model -> {qtype -> {scores, selected, obj_scores}}
    
    for name in FRAME_MODELS:
        df = model_data[name]
        rows = df[df['video_id'] == TARGET_VIDEO_ID]
        if len(rows) == 0:
            print(f"\\n  ⚠️ {name}: no data for {TARGET_VIDEO_ID}")
            continue
        
        model_frame_data[name] = {}
        for _, row in rows.iterrows():
            qt = row['question_type']
            try:
                scores = _json.loads(row['frame_scores']) if pd.notna(row.get('frame_scores')) else None
                selected = _json.loads(row['selected_frame_indices']) if pd.notna(row.get('selected_frame_indices')) else None
                obj_sc = _json.loads(row['obj_scores_per_frame']) if pd.notna(row.get('obj_scores_per_frame')) else None
            except:
                scores, selected, obj_sc = None, None, None
            
            model_frame_data[name][qt] = {
                'frame_scores': scores,
                'selected_frames': selected,
                'obj_scores': obj_sc,
                'question': row['question'],
                'is_correct': row['is_correct'],
                'predicted_answer': row.get('predicted_answer', ''),
                'correct_answer': row.get('correct_answer', ''),
            }
    
    # Print summary of selected frames per model per question type
    for qt in QTYPES:
        qt_short = QTYPE_SHORT[qt]
        print(f"\\n  ── {qt} ({qt_short}) ──")
        for name in FRAME_MODELS:
            if name not in model_frame_data or qt not in model_frame_data[name]:
                continue
            data = model_frame_data[name][qt]
            sel = data['selected_frames']
            correct_icon = '✅' if data['is_correct'] else '❌'
            print(f"    {correct_icon} {name:>12}: selected frames {sel}")
            print(f"                     Q: {data['question'][:80]}")
            print(f"                     Pred: {data['predicted_answer'][:60]}  |  GT: {data['correct_answer'][:60]}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 23: Visualize frame attention scores side by side
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Visualize frame attention scores for each model ──
if not FRAME_MODELS or not model_frame_data:
    print("⚠️ No frame data available. Skipping.")
else:
    # Pick question types that exist for this video
    available_qtypes = []
    for qt in QTYPES:
        if any(qt in model_frame_data.get(m, {}) for m in FRAME_MODELS):
            available_qtypes.append(qt)
    
    n_qt = len(available_qtypes)
    if n_qt == 0:
        print("No question types found for this video.")
    else:
        fig, axes = plt.subplots(n_qt, 1, figsize=(16, 4 * n_qt))
        if n_qt == 1:
            axes = [axes]
        
        for ax, qt in zip(axes, available_qtypes):
            x = np.arange(NUM_FRAMES)
            for name in FRAME_MODELS:
                if name not in model_frame_data or qt not in model_frame_data[name]:
                    continue
                data = model_frame_data[name][qt]
                scores = data['frame_scores']
                if scores is None:
                    continue
                color = MODEL_COLORS.get(name, '#58a6ff')
                ax.bar(x + FRAME_MODELS.index(name) * 0.8 / len(FRAME_MODELS) - 0.4,
                       scores[:NUM_FRAMES], width=0.8/len(FRAME_MODELS),
                       label=name, color=color, alpha=0.8)
                # Mark selected frames
                sel = data['selected_frames']
                if sel:
                    for s in sel:
                        if s < NUM_FRAMES:
                            ax.axvline(x=s, color=color, linewidth=0.8, alpha=0.3, linestyle='--')
            
            ax.set_xticks(x)
            ax.set_xticklabels([f'F{i}' for i in range(NUM_FRAMES)], fontsize=8)
            ax.set_ylabel('Attention Score')
            ax.set_title(f'{QTYPE_SHORT[qt]} — Frame Attention Scores', fontsize=12,
                        fontweight='bold', color='white')
            ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')
        
        fig.suptitle(f'Frame Attention Comparison — {TARGET_VIDEO_ID}',
                     fontsize=15, fontweight='bold', color='white', y=1.01)
        plt.tight_layout()
        plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 24: Show selected frames with attention overlay
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Show the actual selected frames per model with attention info ──
if not FRAME_MODELS or not model_frame_data:
    print("⚠️ No frame data available. Skipping.")
else:
    for qt in available_qtypes:
        print(f"\\n{'='*80}")
        print(f"  Question Type: {qt} ({QTYPE_SHORT[qt]})")
        print(f"{'='*80}")
        
        for name in FRAME_MODELS:
            if name not in model_frame_data or qt not in model_frame_data[name]:
                continue
            data = model_frame_data[name][qt]
            sel = data['selected_frames']
            if sel is None or len(sel) == 0:
                continue
            
            correct_icon = '✅' if data['is_correct'] else '❌'
            print(f"\\n  {correct_icon} {name} — Selected frames: {sel}")
            print(f"     Q: {data['question']}")
            print(f"     Pred: {data['predicted_answer']}")
            print(f"     GT:   {data['correct_answer']}")
            
            n_sel = len(sel)
            fig, axes = plt.subplots(1, n_sel, figsize=(4 * n_sel, 4))
            if n_sel == 1:
                axes = [axes]
            
            frame_scores = data['frame_scores']
            for ax_idx, frame_idx in enumerate(sel):
                ax = axes[ax_idx]
                if frame_idx < len(all_frames):
                    ax.imshow(all_frames[frame_idx])
                else:
                    ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
                
                score_str = f'{frame_scores[frame_idx]:.4f}' if frame_scores and frame_idx < len(frame_scores) else '?'
                ax.set_title(f'Frame {frame_idx}\\nscore={score_str}',
                           fontsize=10, fontweight='bold',
                           color=MODEL_COLORS.get(name, '#58a6ff'))
                ax.axis('off')
            
            fig.suptitle(f'{name} — {QTYPE_SHORT[qt]} — {correct_icon}',
                        fontsize=13, fontweight='bold', color=MODEL_COLORS.get(name, 'white'))
            plt.tight_layout()
            plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 25: Object attention within selected frames
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Object attention scores within selected frames ──
if not FRAME_MODELS or not model_frame_data:
    print("⚠️ No frame data available. Skipping.")
else:
    for qt in available_qtypes:
        print(f"\\n{'='*80}")
        print(f"  OBJECT SCORES — {qt} ({QTYPE_SHORT[qt]})")
        print(f"{'='*80}")
        
        models_with_obj = []
        for name in FRAME_MODELS:
            if name not in model_frame_data or qt not in model_frame_data[name]:
                continue
            data = model_frame_data[name][qt]
            if data['obj_scores'] is not None:
                models_with_obj.append(name)
                obj_sc = data['obj_scores']
                print(f"\\n  {name}:")
                if isinstance(obj_sc, dict):
                    for frame_key, scores in obj_sc.items():
                        print(f"    Frame {frame_key}: {[f'{s:.4f}' for s in (scores if isinstance(scores, list) else [scores])]}")
                elif isinstance(obj_sc, list):
                    for fi, scores in enumerate(obj_sc):
                        if isinstance(scores, list):
                            print(f"    Frame {fi}: {[f'{s:.4f}' for s in scores]}")
                        else:
                            print(f"    Frame {fi}: {scores:.4f}")
        
        if not models_with_obj:
            print("  No object score data available for this question type.")
        else:
            # Bar chart comparison of object scores
            fig, axes = plt.subplots(1, len(models_with_obj), figsize=(7 * len(models_with_obj), 5), squeeze=False)
            for ax_idx, name in enumerate(models_with_obj):
                ax = axes[0][ax_idx]
                data = model_frame_data[name][qt]
                obj_sc = data['obj_scores']
                sel = data['selected_frames'] or []
                
                # Flatten object scores for visualization
                all_obj_scores = []
                all_obj_labels = []
                if isinstance(obj_sc, dict):
                    for fk, scores in obj_sc.items():
                        if isinstance(scores, list):
                            for oi, s in enumerate(scores):
                                all_obj_scores.append(s)
                                all_obj_labels.append(f'F{fk}_O{oi}')
                elif isinstance(obj_sc, list):
                    for fi, scores in enumerate(obj_sc):
                        if isinstance(scores, list):
                            for oi, s in enumerate(scores):
                                all_obj_scores.append(s)
                                all_obj_labels.append(f'F{fi}_O{oi}')
                
                if all_obj_scores:
                    color = MODEL_COLORS.get(name, '#58a6ff')
                    ax.barh(range(len(all_obj_scores)), all_obj_scores,
                           color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
                    ax.set_yticks(range(len(all_obj_labels)))
                    ax.set_yticklabels(all_obj_labels, fontsize=7)
                    ax.set_xlabel('Object Attention Score')
                ax.set_title(f'{name}', fontsize=12, fontweight='bold',
                           color=MODEL_COLORS.get(name, 'white'))
            
            fig.suptitle(f'Object Attention — {QTYPE_SHORT[qt]} — {TARGET_VIDEO_ID}',
                        fontsize=14, fontweight='bold', color='white', y=1.02)
            plt.tight_layout()
            plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 26: Frame overlap analysis across models
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Frame Selection Overlap Analysis (across ALL videos) ──
if not FRAME_MODELS or len(FRAME_MODELS) < 2:
    print("⚠️ Need at least 2 models with frame data. Skipping.")
else:
    print(f"\\n{'='*80}")
    print(f"FRAME SELECTION OVERLAP ANALYSIS")
    print(f"{'='*80}")
    
    from itertools import combinations as _comb
    
    # For each pair of models, compute IoU of selected frames across all shared questions
    for m1, m2 in _comb(FRAME_MODELS, 2):
        df1 = model_data[m1][model_data[m1]['selected_frame_indices'].notna()].copy()
        df2 = model_data[m2][model_data[m2]['selected_frame_indices'].notna()].copy()
        
        # Merge on question key
        KEY = ['video_id', 'question_type', 'question']
        both = df1[KEY + ['selected_frame_indices']].merge(
            df2[KEY + ['selected_frame_indices']], on=KEY, suffixes=('_1', '_2'))
        
        if len(both) == 0:
            print(f"\\n  {m1} vs {m2}: No shared questions with frame data")
            continue
        
        ious = []
        exact_matches = 0
        for _, row in both.iterrows():
            try:
                s1 = set(_json.loads(row['selected_frame_indices_1']))
                s2 = set(_json.loads(row['selected_frame_indices_2']))
                if len(s1 | s2) > 0:
                    iou = len(s1 & s2) / len(s1 | s2)
                    ious.append(iou)
                    if s1 == s2:
                        exact_matches += 1
            except:
                continue
        
        if ious:
            avg_iou = np.mean(ious)
            print(f"\\n  {m1} vs {m2}:")
            print(f"    Shared questions with frame data: {len(ious)}")
            print(f"    Average Frame IoU: {avg_iou:.4f}")
            print(f"    Exact same frames: {exact_matches} ({exact_matches/len(ious)*100:.1f}%)")
            print(f"    IoU distribution: min={min(ious):.3f}, median={np.median(ious):.3f}, max={max(ious):.3f}")
    
    print("\\n🎉 Frame/Object analysis complete!")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN: Section 5
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell(
    "---\n"
    "## 🏆 Part 5: Detailed Comparison vs tranSTR Baseline\n"
    "\n"
    "For each model, compare against **tranSTR** at the video × question-type level:\n"
    "- Which videos/question types did this model **improve** on?\n"
    "- Which videos/question types did this model **regress** on?\n"
    "- Find the ~10 **hardest videos** where ALL models fail the most.",
    md=True
))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 27: Each model vs tranSTR detailed comparison
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Detailed: Each model vs tranSTR ──
BASELINE = 'tranSTR'
assert BASELINE in MODEL_NAMES, f"Baseline '{BASELINE}' not found in {MODEL_NAMES}"

OTHER_MODELS = [m for m in MODEL_NAMES if m != BASELINE]
KEY_COLS_CMP = ['video_id', 'question_type', 'question']

for name in OTHER_MODELS:
    print(f"\\n{'='*80}")
    print(f"  📊 {name} vs {BASELINE}")
    print(f"{'='*80}")
    
    df_base = model_data[BASELINE][KEY_COLS_CMP + ['is_correct', 'predicted_idx']].copy()
    df_other = model_data[name][KEY_COLS_CMP + ['is_correct', 'predicted_idx']].copy()
    df_base = df_base.drop_duplicates(subset=KEY_COLS_CMP)
    df_other = df_other.drop_duplicates(subset=KEY_COLS_CMP)
    
    cmp = df_base.merge(df_other, on=KEY_COLS_CMP, suffixes=('_base', '_other'))
    
    # Categories
    both_correct = cmp[cmp['is_correct_base'] & cmp['is_correct_other']]
    both_wrong = cmp[~cmp['is_correct_base'] & ~cmp['is_correct_other']]
    improved = cmp[~cmp['is_correct_base'] & cmp['is_correct_other']]   # base wrong, other correct
    regressed = cmp[cmp['is_correct_base'] & ~cmp['is_correct_other']]  # base correct, other wrong
    
    print(f"\\n  Total shared questions: {len(cmp):,}")
    print(f"  ✅ Both correct:     {len(both_correct):>6,} ({len(both_correct)/len(cmp)*100:.1f}%)")
    print(f"  ❌ Both wrong:       {len(both_wrong):>6,} ({len(both_wrong)/len(cmp)*100:.1f}%)")
    print(f"  🟢 {name} IMPROVED:  {len(improved):>6,} ({len(improved)/len(cmp)*100:.1f}%)")
    print(f"  🔴 {name} REGRESSED: {len(regressed):>6,} ({len(regressed)/len(cmp)*100:.1f}%)")
    print(f"  Net gain: {len(improved) - len(regressed):+d}")
    
    # Per question type breakdown
    print(f"\\n  Per Question Type:")
    print(f"  {'QType':<25} {'Improved':>10} {'Regressed':>10} {'Net':>8}")
    print(f"  {'-'*55}")
    for qt in QTYPES:
        imp_qt = len(improved[improved['question_type'] == qt])
        reg_qt = len(regressed[regressed['question_type'] == qt])
        net = imp_qt - reg_qt
        net_str = f'{net:+d}'
        print(f"  {qt:<25} {imp_qt:>10} {reg_qt:>10} {net_str:>8}")
    
    # Top videos where this model improved
    if len(improved) > 0:
        imp_by_vid = improved.groupby('video_id').size().sort_values(ascending=False)
        print(f"\\n  🟢 Top videos where {name} IMPROVED over {BASELINE}:")
        for vid, count in imp_by_vid.head(10).items():
            qtypes_imp = improved[improved['video_id'] == vid]['question_type'].tolist()
            print(f"      {vid}: +{count} questions ({', '.join([QTYPE_SHORT.get(q, q) for q in qtypes_imp])})")
    
    # Top videos where this model regressed
    if len(regressed) > 0:
        reg_by_vid = regressed.groupby('video_id').size().sort_values(ascending=False)
        print(f"\\n  🔴 Top videos where {name} REGRESSED vs {BASELINE}:")
        for vid, count in reg_by_vid.head(10).items():
            qtypes_reg = regressed[regressed['video_id'] == vid]['question_type'].tolist()
            print(f"      {vid}: -{count} questions ({', '.join([QTYPE_SHORT.get(q, q) for q in qtypes_reg])})")
    
    # Save detailed comparison to CSV
    improved_out = improved.copy()
    improved_out['change'] = 'improved'
    regressed_out = regressed.copy()
    regressed_out['change'] = 'regressed'
    detail = pd.concat([improved_out, regressed_out], ignore_index=True)
    detail.to_csv(f'comparison_{name}_vs_{BASELINE}.csv', index=False)
    print(f"\\n  💾 Saved comparison_{name}_vs_{BASELINE}.csv ({len(detail)} rows)")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 28: Visualization — improvement/regression per model vs tranSTR
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Visualize improvement/regression per model vs tranSTR ──
n_other = len(OTHER_MODELS)
fig, axes = plt.subplots(1, n_other, figsize=(7 * n_other, 6), squeeze=False)

for ax_idx, name in enumerate(OTHER_MODELS):
    ax = axes[0][ax_idx]
    
    df_base = model_data[BASELINE][KEY_COLS_CMP + ['is_correct']].drop_duplicates(subset=KEY_COLS_CMP)
    df_other = model_data[name][KEY_COLS_CMP + ['is_correct']].drop_duplicates(subset=KEY_COLS_CMP)
    cmp = df_base.merge(df_other, on=KEY_COLS_CMP, suffixes=('_base', '_other'))
    
    imp_counts = []
    reg_counts = []
    for qt in QTYPES:
        sub = cmp[cmp['question_type'] == qt]
        imp_counts.append((~sub['is_correct_base'] & sub['is_correct_other']).sum())
        reg_counts.append((sub['is_correct_base'] & ~sub['is_correct_other']).sum())
    
    x = np.arange(len(QTYPES))
    color = MODEL_COLORS.get(name, '#58a6ff')
    ax.bar(x - 0.2, imp_counts, 0.4, label='Improved', color='#7ee787', alpha=0.85)
    ax.bar(x + 0.2, [-r for r in reg_counts], 0.4, label='Regressed', color='#ff7b72', alpha=0.85)
    ax.axhline(y=0, color='#8b949e', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([QTYPE_SHORT[qt] for qt in QTYPES], fontsize=9)
    ax.set_ylabel('# Questions')
    ax.set_title(f'{name} vs {BASELINE}', fontsize=13, fontweight='bold', color=color)
    ax.legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d')

fig.suptitle(f'Improvement / Regression vs {BASELINE}', fontsize=15, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.show()
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 29: Find ~10 hardest videos where all models fail the most
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(code_cell("""# ── Find the hardest videos: all models fail the most questions ──
print("=" * 80)
print("🔥 HARDEST VIDEOS: Where ALL models struggle the most")
print("=" * 80)

# For each video, count total incorrect across all models and all qtypes
KEY_COLS_V = ['video_id', 'question_type', 'question']
video_fail_counts = {}

for vid in merged['video_id'].unique():
    vid_rows = merged[merged['video_id'] == vid]
    correct_cols = [f'correct_{m}' for m in MODEL_NAMES]
    
    # Count: for each question, how many models got it wrong
    total_wrongs = 0
    total_questions = len(vid_rows)
    for _, row in vid_rows.iterrows():
        n_wrong = sum(1 for col in correct_cols if pd.notna(row[col]) and not row[col])
        total_wrongs += n_wrong
    
    # Also check: how many qtypes have ALL models wrong
    all_wrong_qtypes = 0
    for qt in vid_rows['question_type'].unique():
        qt_rows = vid_rows[vid_rows['question_type'] == qt]
        for _, row in qt_rows.iterrows():
            if all(pd.notna(row[col]) and not row[col] for col in correct_cols):
                all_wrong_qtypes += 1
    
    video_fail_counts[vid] = {
        'total_wrongs': total_wrongs,
        'total_questions': total_questions,
        'all_wrong_qtypes': all_wrong_qtypes,
        'n_models': len(MODEL_NAMES),
        'fail_rate': total_wrongs / (total_questions * len(MODEL_NAMES)) if total_questions > 0 else 0,
    }

df_fails = pd.DataFrame(video_fail_counts).T
df_fails.index.name = 'video_id'

# Strategy: first try videos where ALL models wrong on ALL qtypes
# If not enough, relax to most total wrongs
df_fails = df_fails.sort_values(['all_wrong_qtypes', 'fail_rate'], ascending=[False, False])

# Pick top 10
hardest_videos = df_fails.head(10)

print(f"\\nTop 10 hardest videos (sorted by all-model failures):\\n")
print(f"{'video_id':<40} {'All Wrong':>10} {'Total Wrongs':>12} {'Questions':>10} {'Fail Rate':>10}")
print("-" * 85)
for vid, row in hardest_videos.iterrows():
    print(f"{vid:<40} {int(row['all_wrong_qtypes']):>10} {int(row['total_wrongs']):>12} {int(row['total_questions']):>10} {row['fail_rate']:>9.1%}")

# Show per-model breakdown for these videos
print(f"\\n\\nDetailed breakdown per model for hardest videos:")
for vid, row in hardest_videos.iterrows():
    print(f"\\n  📹 {vid}  (all_wrong={int(row['all_wrong_qtypes'])}, fail_rate={row['fail_rate']:.1%})")
    vid_rows = merged[merged['video_id'] == vid]
    for qt in vid_rows['question_type'].unique():
        qt_rows = vid_rows[vid_rows['question_type'] == qt]
        results = []
        for m in MODEL_NAMES:
            col = f'correct_{m}'
            for _, r in qt_rows.iterrows():
                status = '✅' if (pd.notna(r[col]) and r[col]) else '❌'
                results.append(f"{m}={status}")
        print(f"    {QTYPE_SHORT.get(qt, qt):>3}: {' | '.join(results)}")

# ── Save FULL list sorted by fail rate ──
df_fails.to_csv('all_videos_fail_stats.csv')
print(f"\\n💾 Saved all_videos_fail_stats.csv ({len(df_fails)} videos, full list)")

# ── Filter: videos where ALL models wrong on ALL 6 question types ──
n_qtypes = len(QTYPES)
all6_fail = df_fails[df_fails['all_wrong_qtypes'] >= n_qtypes]
print(f"\\n📊 Videos where ALL models fail ALL {n_qtypes} question types: {len(all6_fail)}")

if len(all6_fail) == 0:
    # Relax: find videos with highest all_wrong_qtypes
    for threshold in range(n_qtypes - 1, 0, -1):
        relaxed = df_fails[df_fails['all_wrong_qtypes'] >= threshold]
        if len(relaxed) > 0:
            print(f"   ↳ Relaxed to >= {threshold} all-wrong qtypes: found {len(relaxed)} videos")
            all6_fail = relaxed
            break

all6_fail.to_csv('videos_all_models_fail.csv')
print(f"💾 Saved videos_all_models_fail.csv ({len(all6_fail)} videos)")

# Also save top 10 separately
hardest_videos.to_csv('hardest_videos_top10.csv')
print(f"💾 Saved hardest_videos_top10.csv")
print("\\n🎯 Use these video_ids in Part 4 to analyze frame/object selections!")
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Build notebook
# ═══════════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

output_path = os.path.join(os.path.dirname(__file__), "evaluation.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Created {output_path}")
print(f"   Total cells: {len(cells)}")
