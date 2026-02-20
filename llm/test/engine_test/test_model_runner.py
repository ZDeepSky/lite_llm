import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from src.engine.model_runner import ModelRunner

def test_ModelRunner():
    QWEN3_PATH="/root/autodl-tmp/qwen3-0.6b"
    model_runner = ModelRunner(QWEN3_PATH)
    assert(model_runner.model_config.to_dict().get("model_type") == "qwen3")