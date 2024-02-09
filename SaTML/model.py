
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)



class LlamaModel(AutoModelForCausalLM):
    def __init__(self, name, base_model_name, device=0, use_mlu=False, use_fp16=False):
        super().__init__(name)



class iFlytekSparkModel():
    def forward():
        pass


class ChatGLMModel():
    def forward():
        pass