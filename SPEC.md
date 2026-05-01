# CHIP v0.2 协议规范

**基于 Day 1 跨 9 个 tokenizer × 200 句 FLORES + 200 个成语 + 45 种标记的 1800+ 行实测数据**

---

## 0. 设计原则

1. **数据驱动**——任何协议元素的选择必须有 token 实测支持,品味分歧让数据裁决
2. **目标自适应**——根据目标 tokenizer 决定启用哪些可选规则
3. **可读 / 可手写 / 可逆**——人类不需要工具也能写出符合协议的 prompt
4. **不依赖 LLM 调用**——所有压缩可纯规则离线完成

## 1. 标签层:`###` 全胜

**实测**:在 9 个 tokenizer 上,`###` 全部 1 token;`<task>` 全部 3 token;`【任务】` 在国产 tokenizer 上 3-4 token,在 mistral 上 5 token。

**结论**:**统一使用 Markdown 风格的 `###` 标签**,不分轨道。

```
### 角色
资深 Python 工程师

### 任务
Code review 下面的代码

### 约束
- 找出 bug
- 给修改建议
- 不修改 license header

### 输入
<c>...用户的代码...</c>

### 输出
JSON,key 用英文 snake_case
{
  "issues": [],
  "suggestions": []
}
```

**为什么不用 `##`?** `##` 也是 1 token,但 `###` 在主流 LLM 训练数据(Markdown 文档)中作为"次级标题"出现频率更高,模型对其的解析更稳定。`##` 在 prompt 中容易被解读为"主标题",和外层 system prompt 冲突。

**为什么不用中文方括号?** 实测它们没有 token 优势,却引入跨平台兼容性风险(部分输入法、部分前端会做全/半角转换)。

## 2. 数据声明字段

每个 CHIP prompt 应在最前(可选)声明:

```
### CHIP-Meta
target: deepseek-v3 | qwen2.5 | gpt-4 | claude-3.5 | universal
features:
  - idiom_compress    # 启用成语压缩(默认在 cl100k 上禁用)
  - lexical           # L1 词法层
  - syntactic         # L2 句法层
budget: 1024
```

**自适应规则**:
- target=universal → 只启用 L1+L2 + 通用词汇,不启用成语压缩
- target=cl100k / o200k → 禁用成语压缩(实测在 OpenAI tokenizer 上成语 ≥4 token,反而增加成本)
- target=baichuan2 / deepseek_v3 → 全功能启用

## 3. 内容层:四层压缩

### 3.1 L1 词法层(全 tokenizer 启用)

啰嗦客套语 → 紧凑动宾。30 条规则已在 `rules.yaml`,例如:
- "请你帮我对 X 进行分析" → "分析 X"
- "也就是说" → "即"
- "因此" → "故"

### 3.2 L2 句法层(全 tokenizer 启用)

模式重排 + 列表化。例如:
- "由于 X 所以 Y" → "X 故 Y"
- "请用 JSON 格式输出" → 直接走协议层 ### 输出 字段

### 3.3 L3 成语层(**仅在国产 tokenizer 上启用**)

基于实测白名单。在不同 target 上启用不同子集:

| Target | 启用成语数 | 来源 |
|---|---|---|
| baichuan2 | 73 | `idiom_whitelist.json["baichuan2"]` |
| deepseek_v3 | 39 | 同上 |
| glm4 | 27 | 同上 |
| qwen2.5 / qwen3 | 10 | 同上 |
| yi | 5 | 同上 |
| **cl100k / o200k / mistral** | **0**(禁用) | 实测 ≥4 token |
| **universal(跨厂商)** | **11**(超级核心集) | 在 ≥3 国产 tokenizer 上 1 token |

**超级核心成语集(universal whitelist)**:
> 全力以赴、众所周知、实事求是、因地制宜、自然而然、精益求精、脚踏实地、持之以恒、循序渐进、家喻户晓、理所当然

这 11 个在 4+ 国产 tokenizer 上都是 1 token,可以**无脑启用**。

**风险提示**:即使在 token 经济性上达标,成语在语义上仍可能引入歧义(参考 Chengyu-Bench 实证 LLM 对成语理解 top-1 仅 ~40%)。CHIP v0.2 默认只启用上面 11 个超级核心成语;扩展到 73 条 baichuan2 白名单需要 Day 4 之后做 Cloze 验证。

### 3.4 L4 协议层(可选,跨 prompt 一致性)

字段命名规范:
- 顶级字段必须用 `### 中文标题`(1 token)
- JSON key 必须英文 snake_case(避免 Pydantic 解析失败)
- 占位符用 `<c>...</c>` 包裹长文本(防 prompt injection)
- 命名实体保留原文,不参与压缩

## 4. 保留不压缩白名单

以下内容**永远不压缩**:
- 数字、日期、版本号、URL
- 代码块(`` ``` `` 包裹)、SQL、JSON、XML
- 用户原话引语("..." 包裹)
- 命名实体(`<ne>` 包裹)
- API key / token 占位符

## 5. 与现有工作的兼容声明

- **LangGPT 兼容**:LangGPT 的 `# Role / ## Profile / ## Skills / ## Rules / ## Workflow` 与 CHIP `### 角色 / ### 任务 / ### 约束 / ### 输出` 同构,可直接互转
- **LLMLingua-2 兼容**:CHIP 输出可作为 LLMLingua-2 的输入,做"协议化 + 神经压缩"的串联

## 6. 版本对比

| 版本 | 主要变化 | 实证支持 |
|---|---|---|
| v0.1 (4/30) | 提出"双轨"标签设计;假设国产 tokenizer 上中文方括号有优势 | 仅基于 §2.3 二手数据 |
| **v0.2 (5/1)** | **统一 `###` 标签**(实测全 tokenizer 1 token);**协议自适应启用成语**(实测白名单) | 9 tokenizer × 1755 idiom + 405 marker 实测 |

CHIP 的协议设计经一次实测后做了重大简化——这是项目数据驱动而非品味驱动的体现。
