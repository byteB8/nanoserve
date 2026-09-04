"""Model shapes, and the arithmetic that decides how much memory a run costs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def n_params(self) -> int:
        """Parameter count, counting the tied lm_head only once."""
        emb = self.vocab_size * self.n_embd + self.block_size * self.n_embd
        per_block = 12 * self.n_embd * self.n_embd + 13 * self.n_embd
        return emb + self.n_layer * per_block + 2 * self.n_embd

    def kv_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        """Bytes of KV-cache a single token occupies, across every layer.

        Two tensors (K and V), one per layer, each n_embd wide:
            2 * n_layer * n_embd * dtype_bytes

        For gpt2 in fp32 this is 2 * 12 * 768 * 4 = 73,728 B (72 KiB) per token.
        Worth internalising: this number, times context length times batch size,
        is what actually caps how many sequences you can serve at once -- and it
        grows with batch while the weights stay fixed.
        """
        return 2 * self.n_layer * self.n_embd * dtype_bytes


PRESETS = {
    "gpt2": GPTConfig(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": GPTConfig(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": GPTConfig(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl": GPTConfig(n_layer=48, n_head=25, n_embd=1600),
}
