<div align="center">

# 🀄 CHIP

**Chinese High-density Instruction Protocol**
中文高密度提示协议

*把啰嗦的中文 prompt 自动压成结构化高密度形式,在保留语义的同时减少 token 消耗*

[![PyPI](https://img.shields.io/pypi/v/chip-prompt?color=blue)](https://pypi.org/project/chip-prompt/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-16%20passing-brightgreen)]()
[![Status](https://img.shields.io/badge/status-v0.1%20MVP-orange)]()
[![Demo](https://img.shields.io/badge/🤗%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/marcuscw/CHIP)

[**English**](README_EN.md)  ·  **中文**  ·  [**SPEC.md**](SPEC.md)  ·  [**CHIP-Bench**](docs/benchmark.md)  ·  [**Roadmap**](docs/roadmap.md)

</div>

---

## 这是什么

CHIP 是一套**协议化、可审计、纯规则**的中文 prompt 压缩器,目标是为 LLM 时代的中文输入提供一套等价于 Markdown 的事实标准。

```python
>>> from chip import compress
>>> compress("请你帮我对下面这段文字进行一个全面的分析,如果可以的话麻烦你给出一些改进的建议")
'分析下面这段文字,给出一些改进的建议'
# 39 字 → 18 字, 字符压缩率 54%
```

跟同类工作的差异:

|                       | LLMLingua-2 | LangGPT | **CHIP** |
|-----------------------|-------------|---------|----------|
| 中文一等公民          | ❌ 零样本迁移 | ⚠️ 无优化 | ✅ |
| 协议化 / 可读 / 可手写 | ❌           | ✅       | ✅ |
| 端到端压缩            | ✅           | ❌(它做扩展)| ✅ |
| 跨厂商 tokenizer 适配 | ⚠️ 需自训   | ✅       | ✅ 双轨 |
| 跑一次的代价          | 神经推理 ~100ms | — | 纯规则 ~5ms |

**项目主张**:

1. **不是"汉语原生更省 token"**——这是经验直觉,在 cl100k 上已被证伪。
2. **是"在国产 LLM 时代,中文 prompt 协议化压缩值得有一套规范"**——填补 LLMLingua 系列与 LangGPT 之间的空白格。
3. **不追求算法新颖性,追求工程可用性 + 社区可采纳性**。

## 安装

```bash
pip install chip-prompt
```

或开发版:

```bash
git clone https://github.com/luancy0423/CHIP
cd CHIP
pip install -e ".[bench,demo]"
```

## 快速上手

### Python

```python
from chip import compress

# 最简单
out = compress("请你帮我总结一下这段文字")

# 带 audit trail
result = compress("请帮我分析", return_result=True)
print(result.diff())
# 原:  请帮我分析
# 压:  分析
# 规则: L1-001×1
```

### CLI

```bash
chip "请你帮我总结一下这段文字"               # 直接出结果
chip "..." --diff                              # 看压缩前后对比
chip --target qwen2.5 --layers L1 L2 "..."     # 指定目标模型和压缩层
echo "..." | chip                              # 管道输入
```

### Web Demo

```bash
pip install chip-prompt[demo]
python scripts/gradio_demo.py
# 打开 http://127.0.0.1:7860
```

## 协议设计要点(v0.1)

CHIP 协议是**双轨**的——根据目标 tokenizer 选不同标签风格:

```
# Qwen / DeepSeek / GLM 轨 (中文方括号在国产 tokenizer 上 1-2 token)
【角】资深 Python 工程师
【任】Code review
【约】
- 找出 bug
- 给修改建议
【出】JSON, key 用英文 snake_case

# cl100k 轨 (英文 XML/Markdown 在 GPT/Claude tokenizer 上 1-3 token)
<role>Senior Python engineer</role>
## Goal
Code review
## Constraints
- Find bugs
- Provide fix suggestions
## Output
JSON, snake_case keys
```

**为什么双轨?** Day 1 实测显示:`【任务】` 在 Qwen2.5 上是 2 token,在 cl100k 上是 5-6 token;`<task>` 反过来。**协议必须是 tokenizer-aware 的**,任何"统一标签"都会牺牲一边。

## 三层压缩架构

```
原始 prompt
    │
    ▼
[L1 词法层] ── 啰嗦套话 → 紧凑动宾,纯正则,~1.3-1.5x
    │           "请你帮我进行分析" → "分析"
    │
    ▼
[L2 句法层] ── 模式重排,需 jieba 分词,~2-3x
    │           "由于X所以Y" → "X故Y"
    │
    ▼
[L3 语义层] ── (Roadmap M3) token classifier 风格,可选,~5-10x
    │
    ▼
压缩后 prompt + audit trail
```

每一层都可以独立开关,每条改动都可追溯到 `rules.yaml` 的某条规则。

## CHIP-Bench (规划中)

Day 4 将发布**首个中文 prompt 压缩公开评测**:
- 任务:CMRC 2018 + DuReader-Long-Zh + C-Eval 子集 + CLUEWSC
- 模型:Qwen2.5-7B/72B、DeepSeek-V3、GLM-4 + (可选) GPT-4o、Claude-3.5
- Baseline:LLMLingua-2、Selective Context、原 prompt
- 指标:压缩率 + 任务准确率 + 语义相似度 (BGE-M3) + 端到端 latency

## 诚实声明

为了避免被"情怀大于实证"批评,以下几点是项目明确**不主张**的:

- ❌ "汉语因为信息密度高所以更省 token"——arXiv:2604.14210 (Mythbuster, 2026) 在 SWE-bench 上证伪
- ❌ "中文 prompt 让模型更聪明"——在英文中心模型上常常相反 (Wendler 2024, Shi 2022)
- ❌ "字本位组合可让 LLM 自动理解新合成词"——arXiv:2410.12656 (Goldman 2024) 给出反向证据
- ❌ "全程使用文言文压缩"——LLM 对生僻文言虚词理解不稳定,易引入幻觉

CHIP 主张的是:**在符合中文训练分布的国产模型上,通过协议化压缩可在 token 经济性、可读性、可审计性三个维度同时提供工程价值**。

## 路线图

- **v0.1 (五一 MVP)**:30 条规则、双轨协议、Gradio demo、Tokenizer baseline
- **v0.2 (M1)**:60+ 规则、CHIP-Bench 公开数据集、与 LLMLingua-2 对照实验
- **v0.3 (M3)**:L3 语义层(基于 Chinese-XLM-R 的 token classifier)
- **v1.0 (M6)**:CHIP-Bench 作为社区标准、长文 paper、agent / function-calling 适配

## 致谢

- 灵感源于哈工大 CSAPP 课上的"汉字编码血泪史"讲述
- 借鉴 [LangGPT](https://github.com/langgptai/LangGPT)(结构化提示协议)与 [LLMLingua](https://github.com/microsoft/LLMLingua)(神经压缩)
- 参考 SCIR 实验室 [Cross-lingual Prompting](https://arxiv.org/abs/2310.14799) (EMNLP 2023) 的多语言 prompt 工程视角

## 引用

```bibtex
@software{chip2025,
  title  = {CHIP: Chinese High-density Instruction Protocol},
  author = {Cui, Wei},
  year   = {2025},
  url    = {https://github.com/marcuscw/CHIP},
  note   = {v0.1, MVP release}
}
```

## License

Apache-2.0
