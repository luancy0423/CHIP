"""
scripts/gradio_demo.py
=======================
CHIP 交互演示。在 HuggingFace Space 部署即可面向公众展示。

跑法 (本地):
    pip install -e ".[demo]"
    python scripts/gradio_demo.py
    打开 http://127.0.0.1:7860

跑法 (HuggingFace Space):
    把项目根目录推到 Space (sdk: gradio),把本文件重命名/复制为 app.py。
"""
from __future__ import annotations
import os, sys
from pathlib import Path

# 让脚本无需 pip install 也能直接跑
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import tiktoken
from chip import Compressor


# 各 tokenizer 加载(轻量,只用 tiktoken;HF tokenizer 太慢加载,demo 不上)
TOKENIZERS = {
    "cl100k (GPT-4)":   tiktoken.get_encoding("cl100k_base"),
    "o200k (GPT-4o)":   tiktoken.get_encoding("o200k_base"),
}

# 注:实际 demo 中可加 Qwen / DeepSeek tokenizer,但要 lazy-load 避免冷启动慢


def count_tokens(text: str, tokenizer_name: str) -> int:
    enc = TOKENIZERS.get(tokenizer_name)
    if enc is None:
        return -1
    return len(enc.encode(text))


def run(text: str, target: str, use_l1: bool, use_l2: bool, tokenizer_name: str):
    if not text.strip():
        return "", "", "", "0 → 0 (—)", "(no rules applied)"

    layers = []
    if use_l1: layers.append("L1")
    if use_l2: layers.append("L2")
    if not layers:
        layers = ["L1"]

    compressor = Compressor(target=target, layers=layers)
    result = compressor.compress(text)

    n_orig = count_tokens(text, tokenizer_name)
    n_comp = count_tokens(result.compressed, tokenizer_name)
    saving_pct = (1 - n_comp / max(n_orig, 1)) * 100
    stats = f"{n_orig} → {n_comp} tokens  ({saving_pct:+.1f}% on {tokenizer_name})"

    rules_str = "\n".join(f"  · {r}" for r in result.applied_rules) or "(no rules matched)"
    return text, result.compressed, stats, stats, rules_str


EXAMPLES = [
    ["请你帮我总结一下这段文字所表达的核心观点是什么", "qwen2.5", True, True, "cl100k (GPT-4)"],
    ["如果可以的话,麻烦你对下面这段代码进行一下分析,并给出一些改进建议", "qwen2.5", True, True, "cl100k (GPT-4)"],
    ["请你扮演一位资深 Python 工程师,对下面的代码进行 code review,并以 JSON 格式输出", "qwen2.5", True, True, "cl100k (GPT-4)"],
    ["因为最近在下雨,所以路面变得很滑,因此开车的时候需要特别注意安全", "qwen2.5", True, True, "cl100k (GPT-4)"],
]


with gr.Blocks(title="CHIP — Chinese High-density Instruction Protocol", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🀄 CHIP · 中文高密度提示协议
    把啰嗦的中文 prompt 自动压成结构化高密度形式,在保留语义的同时减少 token 消耗。

    **当前是 v0.1 早期版本**——L1+L2 共 30 条规则,Day 2-5 持续扩充到 60+。
    """)

    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="原始 prompt", lines=6,
                             placeholder="把你的 prompt 粘进来……")
            with gr.Row():
                target = gr.Dropdown(["qwen2.5", "cl100k", "o200k", "glm4", "deepseek_v3"],
                                     value="qwen2.5", label="目标模型(决定协议轨道)")
                tokenizer_name = gr.Dropdown(list(TOKENIZERS.keys()),
                                             value="cl100k (GPT-4)",
                                             label="计数 tokenizer")
            with gr.Row():
                use_l1 = gr.Checkbox(value=True, label="L1 词法层")
                use_l2 = gr.Checkbox(value=True, label="L2 句法层")
            btn = gr.Button("压缩 →", variant="primary")

        with gr.Column():
            out = gr.Textbox(label="压缩后", lines=6, interactive=False)
            stats = gr.Textbox(label="Token 节省", interactive=False)
            rules_view = gr.Textbox(label="命中的规则(audit trail)",
                                    lines=4, interactive=False)

    gr.Examples(EXAMPLES, [inp, target, use_l1, use_l2, tokenizer_name])

    gr.Markdown("""
    ---
    ### 协议设计要点
    - **双轨**:Qwen 轨用 `【角】【任】` 单字方括号(Qwen tokenizer 上 2 token);
      cl100k 轨用 `<task>` XML(GPT-4 tokenizer 上 3 token)
    - **保护性 mask**:URL、代码块、数字、引语自动跳过,不会被规则误改
    - **可审计**:每次压缩都给出命中的规则 ID,便于追溯
    - **不强制 CoT 语种**:协议只规定输入和输出,不干预模型中间思考过程
      (基于 Wang et al. 2025 的实证)

    📦 GitHub: https://github.com/marcuscw/CHIP   ·  📄 SPEC.md  ·  Apache-2.0
    """)

    btn.click(run, [inp, target, use_l1, use_l2, tokenizer_name],
              [inp, out, stats, stats, rules_view])


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("HOST", "127.0.0.1"),
                server_port=int(os.getenv("PORT", 7860)),
                share=os.getenv("SHARE") == "1")
