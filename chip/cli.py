"""
chip.cli — 命令行入口

安装后:
    chip "请你帮我总结一下这段文字"
    echo "..." | chip
    chip --target qwen2.5 --layers L1 L2 --diff "..."
"""
from __future__ import annotations
import argparse, sys

from chip.compressor import Compressor


def main():
    ap = argparse.ArgumentParser(prog="chip",
        description="CHIP — Chinese High-density Instruction Protocol compressor")
    ap.add_argument("text", nargs="?",
                    help="prompt text (or read from stdin if omitted)")
    ap.add_argument("--target", default="qwen2.5",
                    choices=["qwen2.5", "cl100k", "o200k", "deepseek_v3", "glm4"],
                    help="target tokenizer (decides protocol track)")
    ap.add_argument("--layers", nargs="+", default=["L1", "L2", "L4"],
                    choices=["L1", "L2", "L3", "L4"],
                    help="L1=词法 L2=句法 L3=成语(需国产模型) L4=协议归一化")
    ap.add_argument("--diff", action="store_true",
                    help="show original / compressed / rules side-by-side")
    ap.add_argument("--rules", default=None,
                    help="custom rules.yaml path")
    args = ap.parse_args()

    if args.text:
        text = args.text
    else:
        text = sys.stdin.read()
        if not text.strip():
            ap.print_help()
            sys.exit(1)

    kwargs = {}
    if args.rules:
        kwargs["rules_path"] = args.rules
    compressor = Compressor(target=args.target, layers=args.layers, **kwargs)
    result = compressor.compress(text)

    if args.diff:
        print(result.diff())
        print(f"\n字符压缩率: {result.char_ratio:.2%}  ({len(result.original)} → {len(result.compressed)})")
    else:
        print(result.compressed)


if __name__ == "__main__":
    main()
