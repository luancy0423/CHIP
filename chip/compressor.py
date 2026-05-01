"""
chip.compressor
================
CHIP 主压缩器。设计原则:
    1. 协议是文本,不是模型 — 不依赖 LLM 调用,纯规则可跑
    2. 双轨 — Qwen 轨用中文方括号,cl100k 轨用 XML/Markdown
    3. 可逆 — 保留命名实体、数字、代码、URL 不动
    4. 可审计 — 每条改动可追溯到 rules.yaml 的某条规则

当前实现层级:
    L1 (lex)   — 词法替换:啰嗦套话 → 紧凑动宾,纯正则,~1.3-1.5x 压缩
    L2 (syn)   — 句法重排:虚词替换、列表化,需 jieba 分词,~2-3x
    L3 (sem)   — 语义剪枝(占位,Day 4 之后实现,基于 token-classifier)

Day 2-3 主要工作:扩充 rules.yaml,本文件骨架基本不动。
"""
import os  # ← 新增
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml


# ============ 数据类 ============
@dataclass
class Rule:
    """一条 CHIP 转换规则。"""
    id: str
    layer: str           # "L1" | "L2" | "L3" | "L4"
    pattern: str         # 正则
    replacement: str
    description: str = ""
    saves: int = 0       # 在参考 tokenizer 上预估省多少 token
    risk: str = "low"    # low | mid | high
    flags: int = 0
    _compiled: re.Pattern = field(default=None, repr=False)

    def compile(self):
        if self._compiled is None:
            self._compiled = re.compile(self.pattern, self.flags)
        return self._compiled


@dataclass
class CompressionResult:
    """压缩结果,带 audit trail。"""
    original: str
    compressed: str
    applied_rules: list[str]   # 命中的 rule id 列表
    target: str                # tokenizer 名
    layers: tuple

    @property
    def char_ratio(self) -> float:
        return len(self.compressed) / max(len(self.original), 1)

    def diff(self) -> str:
        """简单的并排展示。"""
        return f"原:  {self.original}\n压:  {self.compressed}\n规则: {', '.join(self.applied_rules) or '(none)'}"


# ============ 规则加载 ============
DEFAULT_RULES_PATH = Path(__file__).parent / "rules" / "rules.yaml"


def load_rules(path: Path | str = DEFAULT_RULES_PATH) -> list[Rule]:
    """从 yaml 加载规则。"""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = []
    for item in data.get("rules", []):
        flags = 0
        for flag_name in item.get("flags", []):
            flags |= getattr(re, flag_name.upper(), 0)
        rules.append(Rule(
            id=item["id"],
            layer=item["layer"],
            pattern=item["pattern"],
            replacement=item.get("replacement", ""),
            description=item.get("description", ""),
            saves=item.get("saves", 0),
            risk=item.get("risk", "low"),
            flags=flags,
        ))
    return rules


# ============ 保护性 mask ============
# 这些 pattern 命中的子串会先被替换成占位符,跑完规则后再还原。
# 防止规则误改专有名词、URL、代码、数字。
PROTECT_PATTERNS = [
    ("URL",   re.compile(r"https?://\S+")),
    ("CODE",  re.compile(r"```[\s\S]*?```|`[^`\n]+`")),
    ("NUM",   re.compile(r"\d+(?:\.\d+)?(?:%|km|kg|m|s|°C)?")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    # 双引号包裹的引文(用户原话)
    ("QUOTE", re.compile(r"[\"\u201c][^\"\u201d]+[\"\u201d]")),
]


# 占位符前缀用一个不会出现在自然中文里、且不会被 PROTECT_PATTERNS 命中的 token
_PH_OPEN = "\u2983"  # ⦃
_PH_CLOSE = "\u2984"  # ⦄
_PH_RE = re.compile(rf"{_PH_OPEN}\d+{_PH_CLOSE}")


def _mask(text: str) -> tuple[str, list[tuple[str, str]]]:
    """把不可压缩片段替换成 ⦃i⦄ 占位符,返回 (masked, mappings)。

    关键:每次 sub 时跳过已经 mask 过的占位符,避免嵌套替换。
    """
    mappings = []
    masked = text

    def make_sub():
        def _sub(m):
            # 如果 match 整体落在已有占位符内,跳过
            content = m.group(0)
            if _PH_RE.fullmatch(content):
                return content
            i = len(mappings)
            placeholder = f"{_PH_OPEN}{i}{_PH_CLOSE}"
            mappings.append((placeholder, content))
            return placeholder
        return _sub

    for tag, pat in PROTECT_PATTERNS:
        masked = pat.sub(make_sub(), masked)
    return masked, mappings


def _unmask(text: str, mappings: list[tuple[str, str]]) -> str:
    # 反向替换避免 ⦃1⦄ 误替换 ⦃10⦄
    for placeholder, original in reversed(mappings):
        text = text.replace(placeholder, original)
    return text


# ============ 主类 ============
class Compressor:
    """可重用的压缩器实例。"""

    def __init__(self,
                 rules_path: Path | str = DEFAULT_RULES_PATH,
                 target: str = "qwen2.5",
                 layers: Iterable[str] = ("L1", "L2", "L4")):
        """
        Args:
            target: 目标 tokenizer,影响成语压缩等 target-aware 决策
            layers: 启用的压缩层
                - L1: 词法层(套话剪枝),保险,默认开
                - L2: 句法层(模式重排),保险,默认开
                - L3: 成语层(语义压缩),需 target 是国产 tokenizer 才有意义,默认关
                - L4: 协议层归一化(### 标题统一),无害,默认开
        """
        self.rules = load_rules(rules_path)
        self.target = target
        self.layers = tuple(layers)
        # 预编译
        for r in self.rules:
            r.compile()

    def compress(self, text: str) -> CompressionResult:
        original = text

        # 可选:jieba 增强角色提取 (pre-process,优先于 L2-022 的纯正则)
        applied_pre = []
        if os.getenv("CHIP_USE_JIEBA") == "1" and "L2" in self.layers:
            text, jieba_applied = _jieba_role_extract(text)
            if jieba_applied:
                applied_pre.append("L2-022J(jieba)")

        masked, mappings = _mask(text)
        applied = list(applied_pre)

        for rule in self.rules:
            if rule.layer not in self.layers:
                continue
            new_text, n = rule._compiled.subn(rule.replacement, masked)
            if n > 0:
                applied.append(f"{rule.id}×{n}")
                masked = new_text

        # 收尾
        masked = re.sub(r"[ \t]+", " ", masked)
        masked = re.sub(r"\s*\n\s*\n\s*\n+", "\n\n", masked)
        # 协议层留下的孤立标点清理
        masked = re.sub(r"\n[,,;;。.\s]+", "\n", masked)
        masked = re.sub(r"^[,,;;]+\s*", "", masked, flags=re.MULTILINE)
        masked = masked.strip()

        compressed = _unmask(masked, mappings)
        return CompressionResult(
            original=original, compressed=compressed,
            applied_rules=applied, target=self.target, layers=self.layers,
        )


# ============ 便捷函数 ============
_default_compressor = None


def compress(text: str,
             target: str = "qwen2.5",
             layers: Iterable[str] = ("L1", "L2", "L4"),
             return_result: bool = False) -> str | CompressionResult:
    """简便入口。

    >>> compress("请帮我总结一下这段文字")
    '总结一下这段文字'

    >>> compress("...", layers=["L1","L2","L3","L4"])  # 启用所有层(包括成语)

    >>> r = compress("...", return_result=True)
    >>> print(r.diff())
    """
    global _default_compressor
    key = (target, tuple(layers))
    if _default_compressor is None or _default_compressor[0] != key:
        _default_compressor = (key, Compressor(target=target, layers=layers))
    result = _default_compressor[1].compress(text)
    return result if return_result else result.compressed

# ============ jieba NP 提取(可选增强) ============
_jieba_loaded = False


def _ensure_jieba():
    """懒加载 jieba。"""
    global _jieba_loaded
    if _jieba_loaded:
        return True
    try:
        import jieba.posseg as pseg  # noqa: F401
        _jieba_loaded = True
        return True
    except ImportError:
        return False


_ROLE_PREFIX_RE = re.compile(
    r"请\s*(?:你)?\s*扮演\s*(?:一(?:个|位))?\s*"
)


def _jieba_role_extract(text: str) -> tuple[str, bool]:
    """用 jieba 词性标注提取最长名词短语作为角色描述。"""
    if not _ensure_jieba():
        return text, False

    import jieba.posseg as pseg

    m = _ROLE_PREFIX_RE.search(text)
    if not m:
        return text, False

    head = text[:m.start()]
    body = text[m.end():]
    if not body:
        return text, False

    words = list(pseg.cut(body))
    np_chars = []
    cumlen = 0
    rest_start = 0
    found_np_core = False

    for w, flag in words:
        is_hard_stop = (
            flag == "w"
            or w in {",", ",", "。", ".", ";", ";", ":", ":", "、", "\n"}
            or flag == "c"
            or flag == "p"
            or (flag.startswith("v") and flag != "vn")
        )
        if is_hard_stop and found_np_core:
            rest_start = cumlen
            break

        np_chars.append(w)
        cumlen += len(w)
        if flag.startswith("n") or flag.startswith("a") or flag == "eng":
            found_np_core = True
    else:
        rest_start = cumlen

    np_str = "".join(np_chars).strip()
    if not np_str or len(np_str) < 2 or not found_np_core:
        return text, False

    rest = body[rest_start:]
    new_text = f"{head}\n### 角色\n{np_str}\n{rest}"
    new_text = re.sub(r"\n[,,;;。.]+", "\n", new_text)
    new_text = new_text.strip()
    return new_text, True