# CHIP 启动清单 — 5 月 1 日上午做完这些

## 0. 环境准备(15 分钟)

```bash
# 拉项目
git clone <你的 repo URL> CHIP
cd CHIP

# 建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 装依赖(开发版)
pip install -e ".[bench,demo]"

# 跑测试看是否一切就绪
pytest -q
# 应该看到:16 passed
```

## 1. 验证压缩器能用(5 分钟)

```bash
chip "请你帮我对下面这段文字进行一个全面的分析" --diff
# 期望输出:
#   原:  请你帮我对下面这段文字进行一个全面的分析
#   压:  分析下面这段文字
#   规则: L1-001×1, L2-001×1
```

如果跑通了,核心引擎就 OK。

## 2. 跑 Day 1 三个实验(2-4 小时,看网速和机器)

### 2.1 Tokenizer baseline

```bash
# 第一次跑会从 HuggingFace 下载 ~10 个 tokenizer,~5GB 磁盘
# 在国内可能需要设置 HF 镜像:export HF_ENDPOINT=https://hf-mirror.com
python scripts/tokenizer_baseline.py --n 200

# 可视化
ls results/
# tokenizer_baseline.csv
# tokenizer_baseline_summary.csv
# zh_en_ratio.png            ← 这张图直接放进 README 第一屏
# chars_per_token.png
```

### 2.2 成语 token 数实测

```bash
python scripts/idiom_test.py
# results/idiom_tokens.csv
# results/idiom_whitelist.json   ← 成语白名单候选
```

### 2.3 标点 / 标签成本

```bash
python scripts/marker_cost.py
# results/marker_cost.csv
# results/marker_cost_heatmap.png  ← 这是第二张关键图
# 同时打印每个 tokenizer 上最便宜的 task-marker
```

### 2.4 内存不够?

如果机器只能加载部分 tokenizer:

```bash
# 只跑国产模型 + cl100k
python scripts/tokenizer_baseline.py --only cl100k qwen2_5 deepseek_v3 glm4 --n 200
```

如果没 GPU 也加载不了 trust_remote_code 的 tokenizer:

```bash
# 最小可行:只跑 OpenAI 系
python scripts/tokenizer_baseline.py --only cl100k o200k --n 500
```

至少这一组数据让你能在 README 里宣称"实测 N 条平行语料"。

## 3. 决定项目主战场(看完 baseline 数据后做)

跑完 §2.1 之后,看 `results/tokenizer_baseline_summary.csv` 的 `zh_per_en_token_mean` 列:

| 情况 | 行动 |
|---|---|
| Qwen / DeepSeek / GLM ratio ≤ 1.05 | ✅ 按计划走,主战场国产模型 |
| 所有 tokenizer ratio > 1.1 | 调整文案,改打"协议化 + 可审计"卖点,不打 token 经济性 |
| 某个 tokenizer 出现 ratio < 0.9 | 🎉 这是你 README 第一句话的素材:"在 X tokenizer 上中文比英文省 X%" |

## 4. Day 1 末尾要交付的东西(对自己交付)

- [ ] `results/` 目录有 5 个 CSV + 2-3 张图
- [ ] 在脑子里(或 commit message 里)能用一句话说出"主战场是 X tokenizer"
- [ ] GitHub repo 已建,push 了 v0.0.1 tag
- [ ] (可选)注册 PyPI 名 `chip-prompt`,上传一个空版本占名

## 5. Day 2 该做什么

打开 `chip/rules/rules.yaml`,把规则从 30 条扩到 60+。重点扩:

- **L1**:再多 15-20 条啰嗦套话剪枝(参考报告 §5.1)
- **L2**:再多 10 条句法重排(参考报告 §5.2)
- **L4**:协议层标签 + meta 字段(参考报告 §5.4)

每加一条规则,在 `tests/test_compressor.py` 里写一条对应测试。

## 6. 已知问题清单(明确,不要花时间在这些 bug 上,标记后下一阶段处理)

- [ ] **L2-022 (扮演角色)** 不能正确处理含空格的复合 NP,如 "资深 Python 工程师"。
      → Day 2 用 jieba 分词 + POS 取最长名词短语。
- [ ] **保护性 mask** 目前不保护"句尾标点紧贴汉字"的情况(罕见)。
- [ ] **L3 语义层** 暂未实现,对应规则文件预留位置。Day 4 之后训 token classifier。
- [ ] **HF Space 部署** 未做,Day 5 上线。

## 7. 如果遇到问题

- **tiktoken HTTP 403**:你的网络访问不到 OpenAI CDN。
  设置代理:`export HTTPS_PROXY=...` 或换镜像
- **HuggingFace 模型下不下来**:`export HF_ENDPOINT=https://hf-mirror.com`
- **trust_remote_code 报错**:某些 tokenizer 需要 `pip install transformers>=4.40`
- **测试在 windows 失败**:中文标点的 unicode 处理,把 `pyproject.toml` 的 python 要求改 3.10+

## 8. 心理预期

- Day 1 结束你应该有:**真实数据 + 可工作的 MVP + 可在终端跑的 demo**
- Day 1 你**不应该**有:漂亮的 paper / 完整的 benchmark / 训好的模型
- 如果 Day 1 跑完发现 token 数据不支持你预期的主张:**调整文案,不要硬凹**
- 如果第一个 commit 后 24 小时没人 star,这是正常的,不焦虑

加油 ✊
