from dataclasses import dataclass
from typing import Optional
from .base_config import BaseConfig

@dataclass
class Qwen3Config(BaseConfig):
    architectures: Optional[list[str]] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 1024
    initializer_range: float = 0.02
    intermediate_size: int = 3072
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    model_type: str = "qwen"
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-06
    rope_scaling: Optional[dict] = None
    rope_theta: int = 1000000
    sliding_window: Optional[int] = None
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936
