"""
scripts/marker_cost.py
=======================
CHIP Day 1 实验 3/3: 标记符号 token 成本测量。

目的:
    决定 CHIP 双轨协议中每条轨道用什么标签。
    cl100k 上 <task> 是 3 token, 【任务】可能 5-6 token (全角字节贵)。
    Qwen 上反过来。这个脚本给出硬数字。

输出:
    results/marker_cost.csv         — marker × tokenizer 的 token 数
    results/marker_cost_heatmap.png — 可视化

跑法:
    python scripts/marker_cost.py
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer_baseline import TOKENIZERS_SPEC


# CHIP 候选标记。分组以便分析。
MARKERS = {
    # 英文/Markdown
    "###":          "md_h3",
    "##":           "md_h2",
    "#":            "md_h1",
    "---":          "md_hr",
    "```":          "md_code",
    "**":           "md_bold",
    # 英文 XML/HTML
    "<task>":       "xml_open",
    "</task>":      "xml_close",
    "<role>":       "xml_open",
    "</role>":      "xml_close",
    "<context>":    "xml_open",
    "</context>":   "xml_close",
    "<doc>":        "xml_short",
    "</doc>":       "xml_short",
    "<c>":          "xml_min",
    "</c>":         "xml_min",
    # 英文方括号(短标签)
    "[T]":          "ascii_short",
    "[R]":          "ascii_short",
    "[Task]":       "ascii_word",
    "[Role]":       "ascii_word",
    # 中文方括号 + 单字
    "【任】":       "zh_min",
    "【角】":       "zh_min",
    "【料】":       "zh_min",
    "【约】":       "zh_min",
    "【流】":       "zh_min",
    "【出】":       "zh_min",
    # 中文方括号 + 双字
    "【任务】":     "zh_word",
    "【角色】":     "zh_word",
    "【上下文】":   "zh_word3",
    "【约束】":     "zh_word",
    "【输出】":     "zh_word",
    # 半角方括号 + 中文
    "[任务]":       "asc_zh_word",
    "[角色]":       "asc_zh_word",
    # 中文标点
    "。":           "zh_punct",
    ",":           "zh_punct",
    ";":           "zh_punct",
    ":":           "zh_punct",
    "「":           "zh_quote",
    "」":           "zh_quote",
    "《":           "zh_quote",
    "》":           "zh_quote",
    # 英文标点
    ".":            "en_punct",
    ",":            "en_punct",
    ";":            "en_punct",
    ":":            "en_punct",
    # 换行/空白
    "\n":           "newline",
    "\n\n":         "newline2",
    " ":            "space",
}


def measure(specs):
    import pandas as pd
    rows = []
    loaded = {}
    for name, family, loader in specs:
        print(f"[LOAD] {name} ...")
        try:
            loaded[name] = (family, loader())
        except Exception as e:
            print(f"[SKIP] {name}: {e}", file=sys.stderr)

    for marker, group in MARKERS.items():
        for name, (family, encode) in loaded.items():
            try:
                ids = encode(marker)
                rows.append({
                    "marker": marker,
                    "marker_repr": repr(marker),
                    "group":  group,
                    "tokenizer": name,
                    "family": family,
                    "n_tokens": len(ids),
                    "ids": str(ids),
                })
            except Exception as e:
                print(f"  [err] {name} on {marker!r}: {e}", file=sys.stderr)

    return pd.DataFrame(rows)


def plot_heatmap(df, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    pivot = df.pivot(index="marker_repr", columns="tokenizer", values="n_tokens")
    fig, ax = plt.subplots(figsize=(11, max(8, len(pivot) * 0.25)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
                cbar_kws={"label": "tokens"}, ax=ax)
    ax.set_title("Marker token cost across tokenizers (lower = cheaper)")
    fig.tight_layout()
    fig.savefig(outdir / "marker_cost_heatmap.png", dpi=150)
    plt.close(fig)


def recommend_track(df) -> str:
    """基于实测数据,自动推荐每个 tokenizer 应该用哪一类标签。"""
    out_lines = ["=== CHIP track recommendation ===\n"]
    for tk in df["tokenizer"].unique():
        sub = df[df["tokenizer"] == tk]
        # 比较 <task> vs 【任务】 vs ### Task
        candidates = {
            "<task>":   sub[sub["marker"] == "<task>"]["n_tokens"].mean(),
            "【任务】": sub[sub["marker"] == "【任务】"]["n_tokens"].mean(),
            "[任务]":   sub[sub["marker"] == "[任务]"]["n_tokens"].mean(),
            "【任】":   sub[sub["marker"] == "【任】"]["n_tokens"].mean(),
            "###":      sub[sub["marker"] == "###"]["n_tokens"].mean(),
        }
        winner = min(candidates, key=candidates.get)
        out_lines.append(
            f"  {tk:<14} cheapest task-marker = {winner!r:>10} "
            f"({candidates[winner]:.1f} tokens) "
            f"| <task>={candidates['<task>']:.0f} 【任务】={candidates['【任务】']:.0f} "
            f"###={candidates['###']:.0f}"
        )
    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    specs = TOKENIZERS_SPEC
    if args.only:
        specs = [s for s in specs if s[0] in set(args.only)]

    df = measure(specs)
    df.to_csv(outdir / "marker_cost.csv", index=False)
    print(f"[INFO] -> {outdir}/marker_cost.csv ({len(df)} rows)")

    print(recommend_track(df))

    try:
        plot_heatmap(df, outdir)
        print(f"[INFO] -> {outdir}/marker_cost_heatmap.png")
    except Exception as e:
        print(f"[WARN] heatmap failed: {e}")


if __name__ == "__main__":
    main()
