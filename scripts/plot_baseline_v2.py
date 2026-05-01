"""
scripts/plot_baseline_v2.py
============================
基于 results/tokenizer_baseline_summary.csv,产出 README 第一屏要用的"主打图"。

跑法:
    python scripts/plot_baseline_v2.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def setup_chinese_font():
    candidates = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
                  "Source Han Sans CN", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] 使用字体: {name}")
            return name
    print("[WARN] 没找到中文字体,标题可能会乱码;请安装 Microsoft YaHei 或 Noto Sans CJK")
    return None


def plot_main(summary: pd.DataFrame, outpath: Path):
    setup_chinese_font()
    summary = summary.sort_values("zh_per_en_token_mean").reset_index(drop=True)

    color_map = {"chinese": "#3B7DD8", "openai": "#E6883C", "international": "#888888"}
    colors = [color_map.get(f, "#888888") for f in summary["family"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(summary["tokenizer"], summary["zh_per_en_token_mean"],
                   color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, summary["zh_per_en_token_mean"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.axvline(1.0, color="#444", ls="--", lw=1.2, alpha=0.7)
    ax.text(1.0, len(summary), " parity (中文=英文)", va="bottom", ha="left",
            fontsize=9, color="#444")

    ax.set_xlabel("ZH-tokens / EN-tokens  (越低 = 中文越省 token)", fontsize=11)
    ax.set_title("中文 vs 英文 token 经济性 — FLORES-200 dev (n=200)\n"
                 "在所有 6 个国产 tokenizer 上,中文 prompt 都不吃亏;在 cl100k 上中文贵 73%",
                 fontsize=12, pad=15)
    ax.set_xlim(0, max(summary["zh_per_en_token_mean"]) * 1.18)

    legend_elements = [
        Patch(facecolor=color_map["chinese"], label="国产 tokenizer"),
        Patch(facecolor=color_map["openai"], label="OpenAI tokenizer"),
        Patch(facecolor=color_map["international"], label="其他国际 tokenizer"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    cn_mean = summary[summary["family"] == "chinese"]["zh_per_en_token_mean"].mean()
    ax.axvline(cn_mean, color=color_map["chinese"], ls=":", lw=1, alpha=0.5)
    ax.text(cn_mean, -0.7, f"国产平均 {cn_mean:.3f}",
            ha="center", fontsize=8, color=color_map["chinese"])

    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 主图已保存到 {outpath}")


def plot_distribution(per_pair: pd.DataFrame, outpath: Path):
    setup_chinese_font()
    order = (per_pair.groupby("tokenizer")["zh_per_en_token"].mean()
             .sort_values().index.tolist())

    family_lookup = (per_pair[["tokenizer", "family"]]
                     .drop_duplicates().set_index("tokenizer")["family"].to_dict())
    color_map = {"chinese": "#3B7DD8", "openai": "#E6883C", "international": "#888888"}

    fig, ax = plt.subplots(figsize=(11, 6))
    data = [per_pair[per_pair["tokenizer"] == t]["zh_per_en_token"].values
            for t in order]

    bp = ax.boxplot(data, vert=True, patch_artist=True, labels=order,
                    showfliers=True, widths=0.6,
                    medianprops=dict(color="black", lw=1.2))
    for patch, t in zip(bp["boxes"], order):
        patch.set_facecolor(color_map.get(family_lookup[t], "#888888"))
        patch.set_alpha(0.85)

    ax.axhline(1.0, color="#444", ls="--", lw=1, alpha=0.6)
    ax.set_ylabel("ZH-tokens / EN-tokens", fontsize=11)
    ax.set_title("每个 tokenizer 上 ZH/EN 比率的分布(箱线图)\n"
                 "OpenAI cl100k 上中文 token 成本 variance 最大 — CHIP 协议化压缩可降低 variance",
                 fontsize=11, pad=12)
    ax.tick_params(axis="x", rotation=20)

    legend_elements = [Patch(facecolor=c, label=f, alpha=0.85)
                       for f, c in color_map.items()]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] 分布图已保存到 {outpath}")


def main():
    outdir = Path("results")
    summary = pd.read_csv(outdir / "tokenizer_baseline_summary.csv")
    per_pair = pd.read_csv(outdir / "tokenizer_baseline.csv")

    plot_main(summary, outdir / "zh_en_ratio_v2.png")
    plot_distribution(per_pair, outdir / "zh_en_ratio_distribution.png")

    print("\n=== README 可以直接抄的表 ===\n")
    print("| Tokenizer | 阵营 | ZH/EN ratio | 中文相对英文 |")
    print("|---|---|---|---|")
    for _, row in summary.sort_values("zh_per_en_token_mean").iterrows():
        camp = {"chinese": "国产", "openai": "OpenAI", "international": "国际"}.get(
            row["family"], row["family"])
        diff = row["zh_per_en_token_mean"] - 1.0
        if diff < 0:
            note = f"中文省 {-diff*100:.1f}%"
        else:
            note = f"中文贵 {diff*100:.1f}%"
        print(f"| **{row['tokenizer']}** | {camp} | {row['zh_per_en_token_mean']:.3f} | {note} |")


if __name__ == "__main__":
    main()