"""Tokenization utilities placeholder."""

from __future__ import annotations


def simple_char_tokenizer(text: str) -> list[int]:
    vocab = {ch: idx for idx, ch in enumerate(sorted(set(text)))}
    return [vocab[ch] for ch in text]
