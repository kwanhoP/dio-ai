from transformers import GPT2LMHeadModel, PretrainedConfig


class GPT2BaseModel(GPT2LMHeadModel):
    name = "gpt2_base"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
