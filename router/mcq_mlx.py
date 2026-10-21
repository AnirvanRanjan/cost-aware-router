"""MLX Option Scorer for multiple-choice questions (DESIGN.md §1.2, briefs/B.md T32).

Reads next-token probabilities for option labels in a single forward pass,
avoiding the formatting collapse ("always A") seen when small models are
prompted to generate an answer letter.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class MLXOptionScorer:
    """Scores multiple-choice option probabilities using next-token logits from an MLX model."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> MLXOptionScorer:
        """Load an MLX model and tokenizer using mlx_lm.load()."""
        try:
            from mlx_lm import load
        except ImportError as e:
            raise ImportError(
                "mlx_lm is required to load MLX models. Note that MLX is only supported on Apple Silicon (macOS arm64)."
            ) from e
        model, tokenizer = load(model_path, **kwargs)
        return cls(model=model, tokenizer=tokenizer)

    def format_prompt(self, prompt: str, enable_thinking: bool = False) -> str:
        """Apply Qwen's chat template with the corpus prompt as the user turn.

        Note: Qwen3-family models have a thinking mode (<think>...</think>).
        For multiple-choice option scoring via next-token logits, thinking mode
        must be disabled (enable_thinking=False) so the next token is an answer letter
        rather than <think>.
        """
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                try:
                    return self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return prompt

    def _tokenize(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "encode"):
            toks = self.tokenizer.encode(text)
            if hasattr(toks, "tolist"):
                return toks.tolist()
            return list(toks)
        return list(self.tokenizer(text))

    def _encode_label_tokens(self, label: str) -> list[int]:
        """Tokenize a label without special tokens."""
        if hasattr(self.tokenizer, "encode"):
            try:
                toks = self.tokenizer.encode(label, add_special_tokens=False)
            except TypeError:
                toks = self.tokenizer.encode(label)
            if hasattr(toks, "tolist"):
                toks = toks.tolist()
            return list(toks)
        return list(self.tokenizer(label))

    def _forward_logits(self, input_ids: list[int]) -> np.ndarray:
        """Run one forward pass and extract logits for the final next-token position."""
        # 1. Try MLX array if mlx is available
        try:
            import mlx.core as mx
            if not isinstance(input_ids, mx.array):
                arr = mx.array([input_ids])
            else:
                arr = input_ids
            out = self.model(arr)
            # Evaluate MLX array if needed
            if hasattr(mx, "eval"):
                mx.eval(out)
            if hasattr(out, "ndim") and out.ndim == 3:
                logits_last = out[0, -1]
            elif hasattr(out, "ndim") and out.ndim == 2:
                logits_last = out[-1]
            else:
                logits_last = out
            return np.asarray(logits_last, dtype=np.float64)
        except (ImportError, Exception):
            pass

        # 2. Try PyTorch tensor if torch is available
        try:
            import torch
            if isinstance(input_ids, list):
                t = torch.tensor([input_ids])
            else:
                t = input_ids
            out = self.model(t)
            if hasattr(out, "logits"):
                out = out.logits
            if hasattr(out, "detach"):
                out = out.detach().cpu().numpy()
            out = np.asarray(out)
            if out.ndim == 3:
                return out[0, -1].astype(np.float64)
            elif out.ndim == 2:
                return out[-1].astype(np.float64)
            return out.astype(np.float64)
        except Exception:
            pass

        # 3. Fallback to generic callable or mock model
        inp = np.asarray([input_ids]) if isinstance(input_ids, list) else input_ids
        out = self.model(inp)
        if hasattr(out, "logits"):
            out = out.logits
        out = np.asarray(out)
        if out.ndim == 3:
            return out[0, -1].astype(np.float64)
        elif out.ndim == 2:
            return out[-1].astype(np.float64)
        return out.astype(np.float64)

    def option_logprobs(
        self, prompt: str, labels: Sequence[str], enable_thinking: bool = False
    ) -> dict[str, float]:
        """Compute the model's next-token log-probabilities for each option label.

        Applies the chat template with `prompt` as the user turn (adding the generation prompt),
        disables thinking mode (`enable_thinking=False`) so Qwen3 does not emit `<think>`,
        runs one forward pass to get next-token logits, and reads each label.
        Takes the max over the label with and without a leading space ("A" vs " A").
        """
        if not labels:
            return {}

        formatted = self.format_prompt(prompt, enable_thinking=enable_thinking)
        input_ids = self._tokenize(formatted)
        next_logits = self._forward_logits(input_ids)

        # Numerically stable log_softmax over the full vocabulary
        max_val = np.max(next_logits)
        logprobs = next_logits - max_val - np.log(np.sum(np.exp(next_logits - max_val)))

        results: dict[str, float] = {}
        for label in labels:
            clean_str = str(label).strip()
            space_str = " " + clean_str

            clean_toks = self._encode_label_tokens(clean_str)
            space_toks = self._encode_label_tokens(space_str)

            lp_clean = float(logprobs[clean_toks[-1]]) if clean_toks else -float("inf")
            lp_space = float(logprobs[space_toks[-1]]) if space_toks else -float("inf")

            results[clean_str] = float(max(lp_clean, lp_space))

        return results
