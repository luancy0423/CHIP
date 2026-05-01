"""
scripts/idiom_test.py
======================
CHIP Day 1 实验 2/3: 四字成语 token 数实测。

目的:
    1. 测量 200 个高频成语在每个 tokenizer 中占多少 token
    2. 筛出 "在目标 tokenizer 上 ≤ 2 token" 的成语 → idiom_whitelist 候选
    3. 验证 §2.4 的"四字成语在 cl100k 上几乎全部被拆成 4 token"假设

输出:
    results/idiom_tokens.csv          — 每成语 × 每 tokenizer 的 token 数
    results/idiom_whitelist.json      — 按 tokenizer 分组的白名单候选

跑法:
    python scripts/idiom_test.py
    python scripts/idiom_test.py --only cl100k qwen2_5

注意:
    白名单只是 token 经济性的初筛,语义可靠性还要 Day 4 用 Chengyu-Bench
    Cloze 任务再过一遍 (≥70% 通过率才能进最终白名单)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 复用 baseline 的 tokenizer 加载逻辑
sys.path.insert(0, str(Path(__file__).parent))
from tokenizer_baseline import TOKENIZERS_SPEC


# 200 个高频成语 (HSK 5/6 + 现代汉语高频)
# 来源:北语 BCC 语料库 + HSK 词表交叉
IDIOMS = """
画蛇添足 亡羊补牢 举一反三 提纲挈领 纲举目张
深思熟虑 抛砖引玉 对症下药 循序渐进 一针见血
切中要害 水到渠成 一气呵成 言简意赅 字斟句酌
点到为止 画龙点睛 大同小异 众所周知 不言而喻
理所当然 显而易见 一目了然 简明扼要 开门见山
直截了当 一语中的 鞭辟入里 入木三分 切中肯綮
迎刃而解 势如破竹 顺理成章 自然而然 顺水推舟
按部就班 有条不紊 井井有条 有的放矢 因地制宜
量体裁衣 因人而异 因时制宜 各有千秋 各有所长
取长补短 集思广益 群策群力 同心协力 齐心协力
精益求精 一丝不苟 兢兢业业 全力以赴 不遗余力
锲而不舍 持之以恒 始终如一 一以贯之 矢志不渝
脚踏实地 实事求是 一丝不挂 因小失大 得不偿失
舍本逐末 缘木求鱼 南辕北辙 背道而驰 适得其反
事与愿违 弄巧成拙 画蛇添足 多此一举 杞人忧天
庸人自扰 杯弓蛇影 草木皆兵 心惊胆战 提心吊胆
忐忑不安 七上八下 心神不宁 坐立不安 寝食难安
高枕无忧 安然无恙 平安无事 风平浪静 一帆风顺
万事如意 心想事成 称心如意 如愿以偿 梦寐以求
朝思暮想 望眼欲穿 翘首以盼 拭目以待 指日可待
为期不远 屈指可数 寥寥无几 凤毛麟角 寥若晨星
比比皆是 不计其数 不胜枚举 数不胜数 多如牛毛
车水马龙 川流不息 络绎不绝 接踵而至 纷至沓来
门庭若市 人山人海 摩肩接踵 水泄不通 人头攒动
门可罗雀 寥寥无几 屈指可数 凤毛麟角 三三两两
鸦雀无声 万籁俱寂 悄无声息 不动声色 默默无闻
鲜为人知 名不见经传 名声赫赫 远近闻名 家喻户晓
妇孺皆知 众人皆知 路人皆知 有口皆碑 交口称赞
赞不绝口 拍案叫绝 叹为观止 叹观止矣 击节称赏
心服口服 心悦诚服 五体投地 顶礼膜拜 推崇备至
望尘莫及 自愧不如 甘拜下风 自惭形秽 相形见绌
黯然失色 大相径庭 截然不同 天壤之别 判若云泥
不可同日而语 不可相提并论 半斤八两 旗鼓相当 平分秋色
不分伯仲 不相上下 势均力敌 棋逢对手 旗鼓相当
难分高下 不分胜负 各有千秋 难解难分 你来我往
针锋相对 剑拔弩张 一触即发 千钧一发 危在旦夕
迫在眉睫 燃眉之急 刻不容缓 当务之急 时不我待
争分夺秒 只争朝夕 分秒必争 风驰电掣 雷厉风行
快马加鞭 马不停蹄 夜以继日 废寝忘食 通宵达旦
""".split()


def measure_idioms(specs):
    import pandas as pd
    rows = []
    loaded = {}
    for name, family, loader in specs:
        print(f"[LOAD] {name} ...")
        try:
            loaded[name] = (family, loader())
        except Exception as e:
            print(f"[SKIP] {name}: {e}", file=sys.stderr)

    for idiom in IDIOMS:
        for name, (family, encode) in loaded.items():
            try:
                tok_ids = encode(idiom)
                rows.append({
                    "idiom": idiom,
                    "tokenizer": name,
                    "family": family,
                    "n_tokens": len(tok_ids),
                    "tokens_repr": str(tok_ids),
                })
            except Exception as e:
                print(f"  [err] {name} on {idiom}: {e}", file=sys.stderr)

    return pd.DataFrame(rows)


def build_whitelist(df, max_tokens: int = 2) -> dict:
    """每个 tokenizer 返回 token 数 ≤ max_tokens 的成语集合。"""
    out = {}
    for tk in df["tokenizer"].unique():
        sub = df[df["tokenizer"] == tk]
        kept = sub[sub["n_tokens"] <= max_tokens]["idiom"].tolist()
        out[tk] = sorted(set(kept))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--max-tokens", type=int, default=2,
                    help="whitelist threshold (token count <= this)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    specs = TOKENIZERS_SPEC
    if args.only:
        specs = [s for s in specs if s[0] in set(args.only)]

    df = measure_idioms(specs)
    df.to_csv(outdir / "idiom_tokens.csv", index=False)
    print(f"[INFO] {len(df)} rows -> {outdir}/idiom_tokens.csv")

    # 分布统计
    print("\n=== Token-count distribution per tokenizer ===")
    pivot = df.groupby(["tokenizer", "n_tokens"]).size().unstack(fill_value=0)
    print(pivot)

    whitelist = build_whitelist(df, args.max_tokens)
    with open(outdir / "idiom_whitelist.json", "w", encoding="utf-8") as f:
        json.dump(whitelist, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Whitelist (<= {args.max_tokens} tokens) -> {outdir}/idiom_whitelist.json")
    for tk, items in whitelist.items():
        print(f"  {tk}: {len(items)} idioms")


if __name__ == "__main__":
    main()
