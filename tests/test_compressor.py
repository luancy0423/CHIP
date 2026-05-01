"""
跑法:
    pip install pytest
    pytest -q
"""
import pytest
from chip import compress, Compressor


# ---------- 基础压缩 ----------
@pytest.mark.parametrize("inp,expected_substr", [
    ("请你帮我总结这段文字",                     "总结"),
    ("如果可以的话,麻烦你分析这段代码",          "分析"),
    ("请对下面的文本进行分析",                   "分析"),
    ("由于下雨所以路滑",                         "故"),
    ("也就是说我们需要重新设计",                 "即"),
    ("请用 JSON 格式输出",                       "### 输出"),  # v0.2: 改为协议字段
])
def test_compresses_to_contain(inp, expected_substr):
    out = compress(inp)
    assert expected_substr in out, f"expected {expected_substr!r} in {out!r}"


def test_compress_makes_shorter():
    long = "请你帮我对下面这段文字进行一个全面的分析,如果可以的话麻烦你给出一些改进的建议"
    short = compress(long)
    assert len(short) < len(long), f"compression failed: {len(short)} >= {len(long)}"


# ---------- 保护性 mask ----------
def test_url_protected():
    inp = "请帮我分析 https://example.com/path?q=1 的内容"
    out = compress(inp)
    assert "https://example.com/path?q=1" in out


def test_code_protected():
    inp = "请帮我看 `x = 1; y = 2` 这段代码"
    out = compress(inp)
    assert "x = 1; y = 2" in out


def test_number_protected():
    inp = "请总结 2024 年第 3 季度的报告"
    out = compress(inp)
    assert "2024" in out and "3" in out


def test_quote_protected():
    inp = '请你解释 "众人皆苦,你也是" 这句话'
    out = compress(inp)
    assert "众人皆苦,你也是" in out


# ---------- audit trail ----------
def test_audit_trail():
    r = compress("请你帮我分析", return_result=True)
    assert len(r.applied_rules) > 0, "expected at least one rule to fire"
    # 至少 L1-001 (请你?帮我?) 命中
    assert any("L1-001" in rid for rid in r.applied_rules)


def test_no_rules_means_unchanged():
    """无规则匹配的 prompt 应原样返回。"""
    r = compress("今天天气不错", return_result=True)
    assert r.compressed.strip() == "今天天气不错"
    assert r.applied_rules == []


# ---------- 层级控制 ----------
def test_l1_only_does_not_run_l2():
    """选一个 L1 不命中、L2 才命中的输入,验证层级开关确实生效。"""
    inp = "由于天气不好所以航班延误"  # L2-003: 由于...所以 → 故
    out_l1_only = compress(inp, layers=["L1"])
    out_with_l2 = compress(inp, layers=["L1", "L2"])
    assert "由于" in out_l1_only, "L1-only 不应消除 '由于'"
    assert "由于" not in out_with_l2, "L1+L2 应消除 '由于'"


# ---------- 幂等性 ----------
def test_idempotent():
    """对同一段已压缩的 prompt 再压缩,结果应基本一致(允许微小差异)。"""
    once = compress("请你帮我分析这段代码")
    twice = compress(once)
    # 长度差异不应超过 2 字符
    assert abs(len(once) - len(twice)) <= 2


# ---------- Compressor 类 ----------
def test_compressor_reusable():
    c = Compressor()
    r1 = c.compress("请帮我总结")
    r2 = c.compress("请帮我分析")
    assert r1.compressed != r2.compressed
    assert all(r.compressed for r in (r1, r2))


# ---------- v0.2 新增:L3 成语层 ----------
def test_l3_idiom_off_by_default():
    """成语层默认关闭(因为它在 cl100k 上反而增加 token)。"""
    out = compress("大家都知道这件事很重要")
    assert "众所周知" not in out, "L3 成语层不应在默认配置下启用"


def test_l3_idiom_on_explicit():
    """显式启用 L3 后,成语压缩应生效。"""
    out = compress("大家都知道这件事很重要", layers=["L1", "L2", "L3", "L4"])
    assert "众所周知" in out


def test_l3_idiom_qwen_target():
    """指定国产 target 时启用 L3 应安全。"""
    out = compress("一步一步地推进项目", target="deepseek_v3",
                   layers=["L1", "L2", "L3", "L4"])
    assert "循序渐进" in out


# ---------- v0.2 新增:L4 协议层归一化 ----------
def test_l4_normalizes_role_tag():
    """非标准的角色标签应归一化为 ### 角色。"""
    out = compress("Role: 资深 Python 工程师")
    assert "### 角色" in out


def test_l4_normalizes_task_tag():
    out = compress("任务:写一个排序算法")
    assert "### 任务" in out


def test_l4_idempotent_on_standard_tags():
    """已经是 ### 任务 的应保持不变。"""
    inp = "### 任务\n写代码"
    out = compress(inp)
    assert "### 任务" in out
    # 不应该被改成 ### 任务\n### 任务\n... 这种重复
    assert out.count("### 任务") == 1


# ---------- 协议层综合测试 ----------
def test_role_play_to_protocol():
    """'请你扮演 X' 应转成 ### 角色 协议字段。"""
    out = compress("请你扮演一位资深的数据分析师", return_result=True)
    assert "### 角色" in out.compressed
