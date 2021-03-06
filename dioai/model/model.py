import re
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    PretrainedConfig,
    RagRetriever,
    RagSequenceForGeneration,
)
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    Seq2SeqLMOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.dpr.configuration_dpr import DPRConfig
from transformers.models.gpt2.modeling_gpt2 import (
    DEPARALLELIZE_DOCSTRING,
    GPT2_INPUTS_DOCSTRING,
    PARALLELIZE_DOCSTRING,
)
from transformers.models.rag.modeling_rag import RetrievAugLMMarginOutput
from transformers.tokenization_utils_base import BatchEncoding

# https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L804
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

# from . import PozalabsModelFactory
from dioai.model.layer import DPROutput
from dioai.preprocessor.encoder.meta import Offset

# from ...train import load_config


class GPT2BaseModel(GPT2LMHeadModel):
    name = "gpt2_base_hf"

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
    name = "gpt2_meta_to_note_hf"


class GPT2ChordMetaToNoteModel(GPT2BaseModel):
    name = "gpt2_chord_meta_to_note_hf"

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

        # ?????? ????????? `num_meta`??? ?????? ?????? (B, 1)????????? ????????? ????????? ??????
        num_meta = num_meta[0].item()
        inputs_embeds = self.transformer.wte(input_ids)
        # ?????? ?????? ????????? ??????
        # chord_progression_vector: 1 or embedding vector
        # ?????? ????????? ????????? ????????? ???????????? ?????????, ?????? ????????? ????????? ??? ??????
        # ?????? ????????? ????????? ????????? ??????
        for idx, (name, info) in enumerate(Offset.__members__.items()):
            if name == "HAS_CHORD_PROGRESSION":
                chord_progression_idx = idx
                has_chord_progression_offset = info.value

        if (
            input_ids.size(1) >= chord_progression_idx + 1
            and (input_ids[:, chord_progression_idx] >= has_chord_progression_offset).all().item()
        ):
            has_chord_progression = input_ids[:, chord_progression_idx]
            # offset ?????? (has_chord_progression: 2????????????, ????????? 0?????? ??????????????? 2??? ???)
            has_chord_progression = has_chord_progression - has_chord_progression_offset
            # TODO: ???????????? ????????? ?????? ?????? ????????? ????????? ??? (574: Yes -> No, 575: No -> Yes)
            # 2021.04.13 ?????? 574??? ?????? ?????? ??????, 575??? ?????? ?????? ???????????? ?????????,
            # Offset 574??? ?????? ?????? ????????? 0, ????????? 1???
            # ???????????? ????????? ????????? ????????? ????????? ???????????? ????????? ????????? ??????????????? ???????????? ?????? ??????
            # ?????? ????????? 1, ????????? 0?????? ??????
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

        # `super().forward()`??? ???????????? ????????? ???????????? ????????? ??? ?????? (?????? ?????? ????????? ????????? ??????)
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
        """`GenerationMixin.prepare_inputs_for_generation` ???????????????"""
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        return {
            **inputs,
            "num_meta": kwargs.get("num_meta"),
            "chord_progression_vector": kwargs.get("chord_progression_vector"),
        }


class BartDenoisingNoteModel(BartForConditionalGeneration):
    name = "bart_denoising_note_hf"

    def __init__(self, config: Union[PretrainedConfig, BartConfig]):
        super().__init__(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class BertForDPR(BertForMaskedLM):
    name = "bert_hf"

    def __init__(self, config: Union[PretrainedConfig, BertConfig]):
        super().__init__(config)
        # for DPR, take the representation at the [CLS] token as the output
        self.bert = BertModel(config, add_pooling_layer=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DPRPretrainedModel(PreTrainedModel):
    config_class = DPRConfig
    load_tf_weights = None
    base_model_prefix = "ctx_encoder"
    _keys_to_ignore_on_load_missing = [r"position_ids"]


DPR_NOTE_OFFSET = 19


class DPRModel(DPRPretrainedModel):
    name = "dpr_model_hf"

    _keys_to_ignore_on_load_missing = [r"position_ids"]
    load_tf_weights = None

    def __init__(self, config: Union[PretrainedConfig, DPRConfig], pretrained_bert):
        super().__init__(config)
        self.config = config
        self.dpr_note_encoder = pretrained_bert
        self.dpr_meta_encoder = pretrained_bert

    def init_weights(self):
        self.dpr_note_encoder.ctx_encoder.init_weights()
        self.dpr_meta_encoder.question_encoder.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        meta = input_ids[:, :DPR_NOTE_OFFSET]
        note = input_ids[:, DPR_NOTE_OFFSET:]
        note_out = self.dpr_note_encoder(
            input_ids=note,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        meta_out = self.dpr_meta_encoder(
            input_ids=meta,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        note_emb = note_out.pooler_output
        meta_emb = meta_out.pooler_output

        sim = torch.matmul(meta_emb, note_emb.T)
        sim = sim.view(meta_emb.size(0), -1)
        softmax_scores = F.log_softmax(sim, dim=1)
        target = torch.arange(0, softmax_scores.size(0)).long().to(softmax_scores.device)
        loss = F.nll_loss(
            softmax_scores,
            target,
            reduction="mean",
        )

        return DPROutput(loss=loss)


RAG_META_END = 19
RAG_NOTE_OFFSET = 20  # DPR?????? ?????? ??? ?????? sos ????????? ???????????? 20?????? ??????


class MusicRagRetriever(RagRetriever):
    """
    RagRetirver??? ?????? ??????, ???????????? meta(question), note(context)??? ???????????? customize
    meta ????????? ???????????? index??? ????????? note ??????????????? ?????? ????????? n_doc??? note??? ?????? meta??? ?????????

    RagGenerator??? ???????????? ??????
    """

    def __init__(self, config, index, init_retrieval=True):
        super().__init__(config, None, None, index=index, init_retrieval=init_retrieval)

    def __call__(
        self,
        question_input_ids: List[List[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified :obj:`question_hidden_states`.
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = question_input_ids

        def generate_square_subsequent_mask(emb_dim):
            """
            attention mask ??????
            """
            mask = (
                torch.triu(
                    torch.ones(emb_dim, (self.config.n_docs) * self.config.retrieval_batch_size)
                )
                == 1
            ).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            return mask

        def postprocess(docs, input_strings, prefix, n_docs):
            """
            meta??? ?????? n_doc??? note??? ?????????
            tensor??? ?????? ??? return
            """

            def _cat_meta_and_note(doc_note, input_meta, prefix):
                if prefix is None:
                    prefix = ""
                out = prefix + input_meta + doc_note
                return out

            def _to_tensor(sequence):
                return torch.tensor(list(map(int, re.findall(r"\d+", str(sequence))))).long()[:-1]

            rag_input_strings = [
                _cat_meta_and_note(
                    docs[i]["text"][j],
                    str(input_strings[i]),
                    prefix,
                )
                for i in range(len(docs))
                for j in range(n_docs)
            ]

            rag_input_tensor = list(map(_to_tensor, rag_input_strings))

            res = None
            for t in rag_input_tensor:
                tmp = t.view(1, -1)
                if res is not None:
                    res = torch.cat([res, tmp])
                else:
                    res = tmp

            return res.long()

        context_input_ids = postprocess(
            docs,
            input_strings,
            prefix,
            n_docs,
        )

        context_attention_mask = generate_square_subsequent_mask(
            self.config.generator.max_position_embeddings
        )

        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "retrieved_doc_embeds": retrieved_doc_embeds,
                "doc_ids": doc_ids,
            },
        )


class MusicRagGenerator(RagSequenceForGeneration):
    name = "musicrag_hf"

    def __init__(self, config, question_encoder, generator, **kwargs):
        self.META_OFFSET = 19

        # retriever
        retriever = MusicRagRetriever(config, index=None)

        super().__init__(
            config=config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever,
            **kwargs,
        )

    def get_nll(
        self,
        seq_logits,
        doc_scores,
        target,
        reduce_loss=True,
        epsilon=0.0,
        exclude_bos_score=True,
        n_docs=None,
    ):
        # shift tokens left
        target = torch.cat(
            [
                target[:, 1:],
                target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id),
            ],
            1,
        )

        n_docs = n_docs if n_docs is not None else self.config.n_docs

        # bos_token_id is None for T5
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1), smooth_obj.squeeze(-1)

        # seq_logits dim = (batch*n_docs, tgt_len , #vocabs)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1)
        )  # batch_size x n_docs x tgt_len x #vocab_size
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)

        # RAG-sequence marginalization
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat(
            [first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2
        )

        # calculate loss
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()

        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)  # total sum of all (normalised) logits

        ll, smooth_obj = _mask_pads(ll, smooth_obj)

        # sum over tokens, exclude bos while scoring
        ll = ll[:, :, 1:].mean(2) if exclude_bos_score and use_bos else ll.mean(2)
        smooth_obj = smooth_obj.mean(2)
        ll = ll.mean(1)  # logsumexp over docs
        smooth_obj = smooth_obj.logsumexp(1)

        nll_loss = -ll
        smooth_loss = -smooth_obj

        if reduce_loss:
            nll_loss = nll_loss.mean()
            smooth_loss = smooth_loss.mean()

        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        exclude_bos_score=None,
        reduce_loss=None,
        labels=None,
        n_docs=None,
        **kwargs,  # needs kwargs for generation
    ):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = (
            exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        )
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        # meta+note??? label??? ??????, dpr?????? ????????? context_input??? ????????? ???????????? ?????? ???
        labels = input_ids
        label_len = labels.size()[1]
        labels = F.pad(
            labels, pad=(0, self.config.generator.max_position_embeddings - label_len), value=0
        )

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        rag_kwargs = {
            "input_ids": input_ids[:, : self.META_OFFSET],
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "context_input_ids": context_input_ids,
            "context_attention_mask": context_attention_mask,
            "doc_scores": doc_scores,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "output_retrieved": output_retrieved,
            "n_docs": n_docs,
        }
        outputs = self.rag(**rag_kwargs)

        loss = None

        if labels is not None:
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=True,
                epsilon=self.config.label_smoothing,
                exclude_bos_score=exclude_bos_score,
                n_docs=n_docs,
            )

        LM_margin_out_dict = {
            "loss": loss,
            "logits": outputs.logits,
            "doc_scores": outputs.doc_scores,
            "past_key_values": outputs.past_key_values,
            "context_input_ids": outputs.context_input_ids,
            "context_attention_mask": outputs.context_attention_mask,
            "retrieved_doc_embeds": outputs.retrieved_doc_embeds,
            "retrieved_doc_ids": outputs.retrieved_doc_ids,
            "question_encoder_last_hidden_state": outputs.question_encoder_last_hidden_state,
            "question_enc_hidden_states": outputs.question_enc_hidden_states,
            "question_enc_attentions": outputs.question_enc_attentions,
            "generator_enc_last_hidden_state": outputs.generator_enc_last_hidden_state,
            "generator_enc_hidden_states": outputs.generator_enc_hidden_states,
            "generator_enc_attentions": outputs.generator_enc_attentions,
            "generator_dec_hidden_states": outputs.generator_dec_hidden_states,
            "generator_dec_attentions": outputs.generator_dec_attentions,
            "generator_cross_attentions": outputs.generator_cross_attentions,
        }

        return RetrievAugLMMarginOutput(**LM_margin_out_dict)
