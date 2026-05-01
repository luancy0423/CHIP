"""
chip — Chinese High-density Instruction Protocol

主要 API:
    >>> from chip import compress
    >>> compress("请你帮我总结一下这段文字的核心观点是什么")
    '总结此文核心观点'

    >>> compress(text, target="qwen2.5", layers=("L1", "L2"))
"""

from chip.compressor import compress, Compressor, CompressionResult

__version__ = "0.1.0"
__all__ = ["compress", "Compressor", "CompressionResult", "__version__"]
