from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config, GPT2LMHeadModel, PretrainedConfig
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import (
    DEPARALLELIZE_DOCSTRING,
    GPT2_INPUTS_DOCSTRING,
    PARALLELIZE_DOCSTRING,
)

# https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L804
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map


class GPT2BaseModel(GPT2LMHeadModel):
    name = "gpt2_base"

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class="GPT2Tokenizer",
        checkpoint="gpt2",
        output_type=CausalLMOutputWithCrossAttentions,
        config_class="GPT2Config",
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GP2MetaToNoteModel(GPT2BaseModel):
    name = "gpt2_meta_to_note"


class GPT2ChordMetaToNoteModel(GPT2BaseModel):
    name = "gpt2_chord_meta_to_note"

    def __init__(self, config: Union[PretrainedConfig, GPT2Config]):
        super().__init__(config)
        self.has_chord_progression_embedding = nn.Embedding(2, config.n_embd)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.has_chord_progression_embedding = self.has_chord_progression_embedding.to(
            self.transformer.first_device
        )
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.has_chord_progression_embedding = self.has_chord_progression_embedding.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class="GPT2Tokenizer",
        checkpoint="gpt2",
        output_type=CausalLMOutputWithCrossAttentions,
        config_class="GPT2Config",
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        chord_progression_vector: torch.Tensor = None,
        num_meta: torch.Tensor = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 평가 시에는 `num_meta`의 형태 또한 (B, 1)이므로 첫번재 요소만 사용
        num_meta = num_meta[0].item()
        inputs_embeds = self.transformer.wte(input_ids)
        # 코드 진행 임베딩 반영
        # chord_progression_vector: 1 or embedding vector
        chord_progression_idx = num_meta - 1

        # 생성 시에는 토큰이 하나씩 생성되기 때문에, 코드 벡터를 적용할 수 없음
        # 최초 입력값 인코딩 시에만 사용
        has_chord_progression_offset = self.config.vocab_size - 2
        if (
            input_ids.size(1) >= num_meta
            and (input_ids[:, chord_progression_idx] >= has_chord_progression_offset).all().item()
        ):
            has_chord_progression = input_ids[:, chord_progression_idx]
            # offset 제거 (has_chord_progression: 2차원이며, 토큰은 0부터 시작하므로 2를 뺌)
            has_chord_progression = has_chord_progression - (self.config.vocab_size - 2)
            # TODO: 불필요한 연산을 막기 위해 전처리 수정할 것 (574: Yes -> No, 575: No -> Yes)
            # 2021.04.13 현재 574가 코드 진행 존재, 575가 코드 진행 부재이기 때문에,
            # Offset 574를 빼면 코드 진행이 0, 부재가 1임
            # 임베딩을 거치기 때문에 그대로 학습에 사용해도 문제가 없지만 의미상으로 명확하게 하기 위해
            # 코드 진행을 1, 부재를 0으로 변환
            has_chord_progression = torch.where(
                has_chord_progression == 0,
                torch.ones_like(has_chord_progression),
                torch.zeros_like(has_chord_progression),
            )
            has_chord_progression_embeds = self.has_chord_progression_embedding(
                has_chord_progression
            )
            chord_progression_embeds = (
                has_chord_progression_embeds * chord_progression_vector
            ) / np.sqrt(self.config.n_embd)
            inputs_embeds[:, chord_progression_idx] = chord_progression_embeds

        # `super().forward()`을 사용하면 원하는 결괏값을 가져올 수 없음 (상속 이슈 때문인 것으로 보임)
        transformer_outputs = self.transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> Dict[str, Any]:
        """`GenerationMixin.prepare_inputs_for_generation` 오버라이딩"""
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        return {
            **inputs,
            "num_meta": kwargs.get("num_meta"),
            "chord_progression_vector": kwargs.get("chord_progression_vector"),
        }
