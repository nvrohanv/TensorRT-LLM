from typing import Optional
from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import (
                                                       DecoderModelForCausalLM,
                                                       register_auto_model)
from tensor
import torch
from torch import nn
from transformers import LlamaConfig
from tensorrt_llm._torch.models.modeling_llama import LlamaModel
import torch.nn.functional as F



class Pooling(torch.nn.Module):
    def __init__(self, pooling_mode: str):
        super().__init__()
        self.pooling_mode = pooling_mode

    def forward(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        pool_type = self.pooling_mode
        if pool_type == "avg":
            epsilon = 1e-9  # A small value to avoid division by zero
            emb = last_hidden.sum(dim=1) / (attention_mask.sum(dim=1)[..., None] + epsilon)
        elif pool_type == "cls":  # tokenizer padding right
            emb = last_hidden[:, 0]
        elif pool_type == "cls__left":  # tokenizer padding left
            seq_idxs = (1 - attention_mask).sum(dim=1)
            batch_size = last_hidden.shape[0]
            batch_idxs = torch.arange(batch_size, device=last_hidden.device)
            emb = last_hidden[batch_idxs, seq_idxs]
        elif pool_type == "last":  # tokenizer padding left
            emb = last_hidden[:, -1]
        elif pool_type == "last__right":  # tokenizer padding right
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        return emb


@register_auto_model("LlamaForTextEmbedding")
class LlamaForTextEmbedding(DecoderModelForCausalLM[LlamaModel,
                                                             LlamaConfig]):
    """
    LlamaForTextEmbedding is a wrapper around the  for text embedding tasks.
    """

    def __init__(self, model_config: ModelConfig[LlamaConfig]):
        nn.Module.__init__(self)
        self.model_config = model_config
        self.model = LlamaModel(model_config,bidirectional=True)

        config = model_config.pretrained_config
        self.pooling = Pooling("avg")


    def forward(self,
                attn_metadata: AttentionMetadata,
                input_ids: torch.LongTensor,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                **kwargs) -> torch.Tensor:
        assert attn_metadata.seq_lens is not None

        hidden_states = self.model(attn_metadata,
                                   input_ids,
                                   position_ids=position_ids,
                                   inputs_embeds=inputs_embeds)
        embeddings = self.pooling_module(hidden_states, inputs["attention_mask"])
        
        dimensions = kwargs.get("dimensions", None)

        if dimensions is not None:
            if not torch.all(dimensions > 0):
                raise ValueError("Dimensions must be positive")

            fill_value = torch.tensor(
                float("-inf"), dtype=embeddings.dtype, device=embeddings.device
            )

            clipped_dimensions = torch.where(
                dimensions < embeddings.shape[1],
                dimensions,
                torch.tensor(embeddings.shape[1], device=embeddings.device),
            )

            embeddings = embeddings.masked_fill(
                torch.arange(embeddings.shape[1], device=embeddings.device)
                >= clipped_dimensions.unsqueeze(-1),
                fill_value,
            )[:, : dimensions.max()]

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings