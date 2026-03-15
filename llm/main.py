
import sys
import os
sys.path.append(os.path.dirname(__file__)+"/src")
# sys.path.append("./code/")

from engine.model_runner import ModelRunner

QWEN3_PATH="/root/autodl-tmp/qwen3-0.6b"


def run():
    model_runner = ModelRunner(QWEN3_PATH)
    text = "你好"
    model_runner.run(text)



if __name__ == "__main__":
    run()