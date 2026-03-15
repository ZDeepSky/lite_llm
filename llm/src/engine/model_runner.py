import json
from pathlib import Path
from typing import Type

from models.qwen3 import Qwen3Config
from transformers import AutoTokenizer
from layers.embedding import liteEmbedding

MAPPING_CLASS:dict[str, Type] ={
    "qwen3": Qwen3Config
}

class ModelRunner:
    def __init__(self, checkpoint_dir: str):
        # 模型路径
        self.checkpoint_dir = checkpoint_dir
        # 模型配置加载
        config_path = Path(checkpoint_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"[config_path]:{config_path} not found")

        self.model_config = self._load_model_config(config_path)
        print(f"{self.model_config}")
        print(f"{self.model_config.model_type}")
        # # tokenizer加载 先试用HF的
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_dir
        )
        # # 加载embedding
        self.emb = liteEmbedding(
            self.model_config.vocab_size,
            self.model_config.hidden_size
        )

        # 模型权重加载


    def _load_model_config(self, config_path: Path)->dict:
        paras = json.loads(config_path.read_text())
        print(f"{paras["model_type"].lower()}")
        cfg_cls = MAPPING_CLASS.get(paras["model_type"].lower())
        if not cfg_cls:
            raise ValueError(f"Unsupported model_type {paras["model_type"]}")

        cfg_cls = cfg_cls.from_dict(paras)
        return cfg_cls

    def run(self, text:str):
        #token化
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"]

        #embedding
        emb = self.emb(input_ids)

        print("Embedding output shape:", emb.shape)
        return emb