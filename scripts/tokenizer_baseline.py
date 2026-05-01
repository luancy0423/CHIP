"""
scripts/tokenizer_baseline.py
===============================
CHIP Day 1 实验 1/3: 中英文 token 比率基线测量。

目的:在 N (默认 200) 条 FLORES-200 zho_Hans-eng_Latn 平行句对上,
测量每个 tokenizer 的 ZH-tokens / EN-tokens 比率。这是 §2.2 所有
主张的事实根据。

输出:
    results/tokenizer_baseline.csv
    results/tokenizer_baseline_summary.csv
    results/zh_en_ratio.png   (条形图)
    results/chars_per_token.png

跑法:
    pip install -e ".[bench]"
    python scripts/tokenizer_baseline.py --n 200

环境:
    - 网络要能访问 huggingface.co (下载 tokenizer)
    - 第一次跑会下 ~10 个 tokenizer,~5GB 磁盘
    - tiktoken 需要从 OpenAI CDN 下 BPE 表,如果挂代理,设 HTTPS_PROXY
    - HF 国内访问慢可设 HF_ENDPOINT=https://hf-mirror.com

控制内存:
    脚本只加载 tokenizer (不加载完整模型权重),11 个 tokenizer 约占 1-2GB RAM。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

# ---- 1. 注册所有 tokenizer ----
TOKENIZERS_SPEC = [
    # OpenAI / 国际
    ("cl100k",     "openai",     lambda: _load_tiktoken("cl100k_base")),
    ("o200k",      "openai",     lambda: _load_tiktoken("o200k_base")),
    ("llama3",     "international", lambda: _load_hf("meta-llama/Meta-Llama-3-8B")),
    ("mistral_v3", "international", lambda: _load_hf("mistralai/Mistral-7B-Instruct-v0.3")),
    # 国产
    ("qwen2_5",    "chinese",    lambda: _load_hf("Qwen/Qwen2.5-7B")),
    ("qwen3",      "chinese",    lambda: _load_hf("Qwen/Qwen3-8B")),
    ("deepseek_v3","chinese",    lambda: _load_hf("deepseek-ai/DeepSeek-V3")),
    ("glm4",       "chinese",    lambda: _load_hf("THUDM/glm-4-9b-chat")),
    ("yi",         "chinese",    lambda: _load_hf("01-ai/Yi-1.5-9B")),
    ("baichuan2",  "chinese",    lambda: _load_hf("baichuan-inc/Baichuan2-7B-Base")),
    ("internlm2",  "chinese",    lambda: _load_hf("internlm/internlm2-7b")),
]


def _load_tiktoken(name: str) -> Callable[[str], list]:
    import tiktoken
    enc = tiktoken.get_encoding(name)
    return lambda text: enc.encode(text)


def _load_hf(model_id: str) -> Callable[[str], list]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    def encode(text: str):
        return tok.encode(text, add_special_tokens=False)
    return encode


# ---- 2. 加载平行语料 ----
def load_parallel_corpus(n: int) -> list[tuple[str, str]]:
    """加载中英平行句对。

    HF datasets v3+ 不再支持 loading script,FLORES 旧版 ('facebook/flores',
    'Muennighoff/flores200') 全部失效。这里按以下顺序尝试:
        1. mteb/flores (Parquet, 无需登录,中英都在 default split)
        2. Helsinki-NLP/opus-100 (zh-en, 100k 对,Parquet)
        3. 本地兜底 (BUILTIN_PAIRS, 50 对) — 网络全挂时仍能跑出一份小基线

    返回 [(zh, en), ...] N 对。
    """
    from datasets import load_dataset

    # 尝试 1: mteb/flores —— 这是社区维护的纯 Parquet 镜像
    try:
        ds = load_dataset("mteb/flores", "zho_Hans-eng_Latn", split="dev")
        pairs = [(ex["sentence1"], ex["sentence2"])
                 for ex in ds.select(range(min(n, len(ds))))]
        print(f"[INFO] Loaded {len(pairs)} pairs from mteb/flores")
        return pairs
    except Exception as e:
        print(f"[WARN] mteb/flores failed: {e}", file=sys.stderr)

    # 尝试 2: opus-100 —— UN 文档为主,质量一般但量大且无登录
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "en-zh", split="test")
        pairs = []
        for ex in ds:
            tr = ex["translation"]
            pairs.append((tr["zh"], tr["en"]))
            if len(pairs) >= n:
                break
        print(f"[INFO] Loaded {len(pairs)} pairs from Helsinki-NLP/opus-100")
        return pairs
    except Exception as e:
        print(f"[WARN] opus-100 failed: {e}", file=sys.stderr)

    # 尝试 3: 兜底 —— 内置 50 对句子,网络全挂也能跑
    print(f"[WARN] Network corpora unavailable, falling back to {len(BUILTIN_PAIRS)} built-in pairs")
    return BUILTIN_PAIRS[:n]


# 内置兜底语料:50 对中英句子,覆盖日常、技术、新闻、文学四个领域
# 网络数据源全挂时,至少这一份能让 baseline 出图
BUILTIN_PAIRS = [
    # 日常
    ("今天天气很好,我打算去公园散步。", "The weather is nice today, I plan to take a walk in the park."),
    ("请问最近的地铁站在哪里?", "Where is the nearest subway station?"),
    ("我想点一杯咖啡和一个三明治。", "I would like to order a coffee and a sandwich."),
    ("这家餐厅的菜很好吃,但价格有点贵。", "The food at this restaurant is delicious but a bit expensive."),
    ("明天我们一起去看电影怎么样?", "How about we go to watch a movie together tomorrow?"),
    ("我已经在这家公司工作了五年。", "I have been working at this company for five years."),
    ("孩子们正在花园里玩耍。", "The children are playing in the garden."),
    ("请把窗户关上,外面风很大。", "Please close the window, it is windy outside."),
    ("我喜欢在周末读书和听音乐。", "I like to read books and listen to music on weekends."),
    ("这本书是我从图书馆借的。", "I borrowed this book from the library."),
    # 技术
    ("人工智能正在改变我们的生活方式。", "Artificial intelligence is changing the way we live."),
    ("请检查这段代码中是否有错误。", "Please check whether there are any errors in this code."),
    ("数据库连接超时,请稍后再试。", "The database connection timed out, please try again later."),
    ("这个算法的时间复杂度是多少?", "What is the time complexity of this algorithm?"),
    ("我们需要优化这个模型的性能。", "We need to optimize the performance of this model."),
    ("机器学习需要大量的训练数据。", "Machine learning requires a large amount of training data."),
    ("请将文件保存为 PDF 格式。", "Please save the file in PDF format."),
    ("这个函数的返回值类型是字符串。", "The return type of this function is a string."),
    ("分布式系统的一致性是一个难题。", "Consistency in distributed systems is a difficult problem."),
    ("神经网络可以识别图像中的物体。", "Neural networks can recognize objects in images."),
    # 新闻
    ("中国和欧盟签署了新的贸易协议。", "China and the European Union signed a new trade agreement."),
    ("研究人员发现了一种新的疾病疗法。", "Researchers have discovered a new treatment for the disease."),
    ("世界经济正在缓慢复苏。", "The world economy is slowly recovering."),
    ("政府宣布了一项新的环保政策。", "The government announced a new environmental policy."),
    ("这次地震造成了重大损失。", "The earthquake caused significant damage."),
    ("奥运会将于明年在巴黎举行。", "The Olympic Games will be held in Paris next year."),
    ("公司今年的利润比去年增长了百分之二十。", "The company's profit grew by 20 percent compared to last year."),
    ("教育部宣布了高考改革方案。", "The Ministry of Education announced a college entrance exam reform plan."),
    ("科学家成功发射了一颗新的人造卫星。", "Scientists successfully launched a new artificial satellite."),
    ("这位作家获得了今年的诺贝尔文学奖。", "This author won the Nobel Prize in Literature this year."),
    # 文学/正式
    ("时间是世界上最公平的东西。", "Time is the fairest thing in the world."),
    ("一千个读者眼里有一千个哈姆雷特。", "Every reader has their own Hamlet."),
    ("书籍是人类进步的阶梯。", "Books are the stepping stones of human progress."),
    ("生活中最重要的不是结果,而是过程。", "The most important thing in life is not the result, but the process."),
    ("失败是成功之母。", "Failure is the mother of success."),
    ("知识改变命运。", "Knowledge changes fate."),
    ("每一个不曾起舞的日子都是对生命的辜负。", "Every day not spent dancing is a betrayal of life."),
    ("沉默有时比千言万语更有力量。", "Silence is sometimes more powerful than a thousand words."),
    ("真正的朋友会在你最需要的时候出现。", "A true friend appears when you need them most."),
    ("世界上没有永远的朋友,只有永远的利益。", "There are no eternal friends in the world, only eternal interests."),
    # 长句
    ("尽管面临许多困难,他始终没有放弃自己的梦想,最终在多年的努力之后取得了成功。",
     "Despite facing many difficulties, he never gave up on his dream, and eventually succeeded after years of effort."),
    ("这项研究表明,经常运动的人比不运动的人在心理健康方面有更好的表现。",
     "This study shows that people who exercise regularly perform better in mental health than those who do not."),
    ("如果你想真正掌握一门语言,仅仅记住单词和语法是不够的,还需要大量的阅读和交流。",
     "If you want to truly master a language, just memorizing words and grammar is not enough; extensive reading and communication are also needed."),
    ("公司的领导层经过长时间的讨论,最终决定将总部从北京搬迁到上海。",
     "After lengthy discussions, the company's leadership ultimately decided to move the headquarters from Beijing to Shanghai."),
    ("这部电影通过一个普通家庭的故事,深刻地反映了当代社会的种种问题。",
     "Through the story of an ordinary family, this film profoundly reflects various issues in contemporary society."),
    ("随着技术的不断发展,远程办公已经成为越来越多公司的常态。",
     "With the continuous development of technology, remote work has become the norm for more and more companies."),
    ("尽管经济形势严峻,但许多创业者仍然对未来充满信心,继续投入到新的项目中。",
     "Despite the severe economic situation, many entrepreneurs are still full of confidence in the future and continue to invest in new projects."),
    ("研究人员花费了五年时间收集数据,最终得出了一个令人意想不到的结论。",
     "Researchers spent five years collecting data and ultimately reached an unexpected conclusion."),
    ("这本小说讲述了一个发生在二十世纪初的爱情故事,情节扣人心弦,人物刻画生动。",
     "This novel tells a love story that took place in the early twentieth century, with a gripping plot and vivid characterization."),
    ("在全球化的背景下,各国之间的文化交流变得越来越频繁,也越来越重要。",
     "In the context of globalization, cultural exchanges between countries have become more frequent and more important."),
]


def load_local_corpus(path: str) -> list[tuple[str, str]]:
    """本地 TSV: zh<TAB>en 一行一对。"""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            zh, en = line.split("\t", 1)
            pairs.append((zh, en))
    return pairs


# ---- 3. 主测量 ----
def measure(pairs: list[tuple[str, str]],
            tokenizer_specs: list[tuple]) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    loaded = {}
    for name, family, loader in tokenizer_specs:
        print(f"[LOAD] {name} ({family}) ...")
        try:
            loaded[name] = (family, loader())
        except Exception as e:
            print(f"[SKIP] {name}: {e}", file=sys.stderr)
            continue

    print(f"[INFO] Successfully loaded {len(loaded)} / {len(tokenizer_specs)} tokenizers")
    print(f"[INFO] Measuring on {len(pairs)} sentence pairs ...")

    for i, (zh, en) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  ... pair {i}/{len(pairs)}")
        for name, (family, encode) in loaded.items():
            try:
                zh_tok = len(encode(zh))
                en_tok = len(encode(en))
                rows.append({
                    "tokenizer": name,
                    "family":    family,
                    "pair_id":   i,
                    "zh_chars":  len(zh),
                    "en_chars":  len(en),
                    "en_words":  len(en.split()),
                    "zh_tokens": zh_tok,
                    "en_tokens": en_tok,
                    "zh_per_en_token":  zh_tok / max(en_tok, 1),
                    "zh_chars_per_tok": len(zh) / max(zh_tok, 1),
                    "en_chars_per_tok": len(en) / max(en_tok, 1),
                })
            except Exception as e:
                print(f"  [err] {name} on pair {i}: {e}", file=sys.stderr)

    return pd.DataFrame(rows)


def summarize(df: "pd.DataFrame") -> "pd.DataFrame":
    summary = df.groupby(["tokenizer", "family"]).agg(
        n_pairs=("pair_id", "count"),
        zh_per_en_token_mean=("zh_per_en_token", "mean"),
        zh_per_en_token_median=("zh_per_en_token", "median"),
        zh_chars_per_tok_mean=("zh_chars_per_tok", "mean"),
        en_chars_per_tok_mean=("en_chars_per_tok", "mean"),
    ).round(3).reset_index()
    summary = summary.sort_values("zh_per_en_token_mean")
    return summary


def plot(summary: "pd.DataFrame", outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    # 图 1: ZH/EN ratio 横向条形
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0" if f == "chinese" else "#DD8452"
              for f in summary["family"]]
    ax.barh(summary["tokenizer"], summary["zh_per_en_token_mean"], color=colors)
    ax.axvline(1.0, color="gray", ls="--", lw=1, label="parity")
    ax.set_xlabel("ZH-tokens / EN-tokens (lower = ZH cheaper)")
    ax.set_title("Chinese vs English token ratio across tokenizers\n(FLORES-200 dev)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "zh_en_ratio.png", dpi=150)
    plt.close(fig)

    # 图 2: chars/token 双轴
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(summary))
    w = 0.35
    ax.bar([i - w/2 for i in x], summary["zh_chars_per_tok_mean"], w,
           label="ZH chars/tok", color="#4C72B0")
    ax.bar([i + w/2 for i in x], summary["en_chars_per_tok_mean"], w,
           label="EN chars/tok", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary["tokenizer"], rotation=30, ha="right")
    ax.set_ylabel("chars per token (higher = more efficient)")
    ax.set_title("Tokenizer efficiency: characters per token")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "chars_per_token.png", dpi=150)
    plt.close(fig)

    print(f"[INFO] Plots saved to {outdir}/")


# ---- 4. CLI ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="number of FLORES sentence pairs to measure on")
    ap.add_argument("--corpus", type=str, default=None,
                    help="optional local TSV (zh<TAB>en per line); overrides FLORES")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--only", type=str, nargs="*", default=None,
                    help="subset of tokenizer names, e.g. cl100k qwen2_5")
    ap.add_argument("--skip-plot", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. corpus
    if args.corpus:
        pairs = load_local_corpus(args.corpus)[:args.n]
    else:
        pairs = load_parallel_corpus(args.n)

    # 2. tokenizers
    specs = TOKENIZERS_SPEC
    if args.only:
        specs = [s for s in specs if s[0] in set(args.only)]
        print(f"[INFO] Filtered to {len(specs)} tokenizers: {[s[0] for s in specs]}")

    # 3. measure
    df = measure(pairs, specs)
    df.to_csv(outdir / "tokenizer_baseline.csv", index=False)
    print(f"[INFO] Per-pair results -> {outdir}/tokenizer_baseline.csv ({len(df)} rows)")

    # 4. summary
    summary = summarize(df)
    summary.to_csv(outdir / "tokenizer_baseline_summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    # 5. plot
    if not args.skip_plot:
        try:
            plot(summary, outdir)
        except Exception as e:
            print(f"[WARN] plotting failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()