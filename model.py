from typing import Optional, Dict, Any
from transformers import T5ForConditionalGeneration, T5Config, GenerationConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DaaType


class DockingAwareAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, daa_type=DaaType.ALL):
        super(DockingAwareAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.daa_type = daa_type

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        if daa_type == DaaType.ATTENTION or daa_type == DaaType.ALL:
            self.q_proj = nn.Linear(input_dim, input_dim)
            self.k_proj = nn.Linear(input_dim, input_dim)
            self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        self.empty_emb = nn.Embedding(1, output_dim)

    def replace_empty_emb(self, x, docking_scores):
        empty_mask = docking_scores.sum(dim=1) == 0  # (batch_size)
        empty_mask = empty_mask.unsqueeze(1)  # (batch_size, 1)
        empty_emb = self.empty_emb(torch.tensor([0], device=x.device))  # (1, input_dim)
        empty_emb = empty_emb.unsqueeze(0)  # (1, 1, input_dim)
        empty_emb = empty_emb.expand(x.size(0), 1, -1)  # (batch_size, 1, input_dim)
        x = torch.where(empty_mask.unsqueeze(-1), empty_emb, x)
        return x

    def _forward_mean(self, x, docking_scores, mask=None):
        return x.mean(dim=1).unsqueeze(1)

    def _forward_docking(self, x, docking_scores, mask=None):

        docking_scores = docking_scores / docking_scores.sum(dim=1, keepdim=True)  # (batch_size, seq_len)
        docking_scores = docking_scores.unsqueeze(-1)  # (batch_size, seq_len, 1)
        if mask is not None:
            d_mask = mask.bool().unsqueeze(-1)
            docking_scores = docking_scores.masked_fill(~d_mask, 0)
        return (docking_scores * x).sum(dim=1).unsqueeze(1)

    def _forward_attention(self, x, docking_scores, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_mask = mask.bool().unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)
        return context[:, 0, :].unsqueeze(1)

    def _forward(self, x, docking_scores, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            docking_scores: Tensor of shape (batch_size, seq_len)
            mask: Optional tensor for masking (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        x_mean = self._forward_mean(x, docking_scores, mask)
        if self.daa_type == DaaType.MEAN:
            return x_mean
        x_docking = self._forward_docking(x, docking_scores, mask)
        if self.daa_type == DaaType.DOCKING:
            return self.alpha * x_mean + x_docking
        x_attention = self._forward_attention(x, docking_scores, mask)
        if self.daa_type == DaaType.ATTENTION:
            return x_attention
        # ALL
        return self.alpha * x_mean + self.beta * x_docking + x_attention

    def forward(self, x, docking_scores, mask=None):
        res = self._forward(x, docking_scores, mask)
        res = self.out_proj(res)
        res = self.replace_empty_emb(res, docking_scores)
        return res


class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, daa_type, prot_dim=2560, emb_dropout=0.0):
        super(CustomT5Model, self).__init__(config)
        self.daa_type = DaaType(daa_type)
        if self.daa_type != DaaType.NO:
            self.docking_attention = DockingAwareAttention(prot_dim, config.d_model, config.num_heads, self.daa_type)
            self.emb_dropout = emb_dropout
            if self.emb_dropout > 0:
                self.emb_dropout_layer = nn.Dropout(emb_dropout)

    def prep_input_embeddings(self, input_ids, attention_mask, emb, emb_mask, docking_scores):
        input_embeddings = self.shared(input_ids)  # Shape: (batch_size, sequence_length, embedding_dim)
        if self.daa_type == DaaType.NO:
            return input_embeddings, attention_mask
        batch_size, seq_length, emb_dim = input_embeddings.shape
        emb = self.docking_attention(emb, docking_scores, mask=emb_mask)[:, 0]  # CLS token
        emb = emb.unsqueeze(1)
        if self.emb_dropout > 0:
            emb = self.emb_dropout_layer(emb)
        new_input_embeddings = torch.cat([emb, input_embeddings], dim=1)
        emb_attention = torch.ones(batch_size, emb.shape[1], device=attention_mask.device)
        attention_mask = torch.cat([emb_attention, attention_mask], dim=1)
        return new_input_embeddings, attention_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, encoder_outputs=None,
                emb=None, emb_mask=None, docking_scores=None, **kwargs):
        if encoder_outputs is not None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   encoder_outputs=encoder_outputs, **kwargs)

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.prep_input_embeddings(input_ids, attention_mask, emb, emb_mask,
                                                                       docking_scores)
        output = super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        return output

    def _prepare_encoder_decoder_kwargs_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            model_kwargs,
            model_input_name: Optional[str],
            generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(self.config)
        inputs_embeds, model_kwargs["attention_mask"] = self.prep_input_embeddings(inputs_tensor,
                                                                                   model_kwargs["attention_mask"],
                                                                                   model_kwargs["emb"],
                                                                                   model_kwargs["emb_mask"],
                                                                                   model_kwargs["docking_scores"])
        model_kwargs["inputs_embeds"] = inputs_embeds
        return super()._prepare_encoder_decoder_kwargs_for_generation(
            None, model_kwargs, model_input_name, generation_config
        )


if __name__ == "__main__":
    # Test the model
    from transformers import PreTrainedTokenizerFast
    from preprocessing.tokenizer_utils import TOKENIZER_DIR, get_ec_tokens

    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    new_tokens = get_ec_tokens()
    tokenizer.add_tokens(new_tokens)
    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      decoder_start_token_id=tokenizer.pad_token_id)
    # for daa_type in [0, 1, 2, 3]:
    #     for lin_attn in [0, 1]:
    for daa_type in [0, 1, 2, 3, 4]:

        print(daa_type)
        model = CustomT5Model(config, daa_type)
        # print number of parameters
        n1 = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters:{n1:,}")
        # number of params in the docking_attention submodule
        if daa_type == 0:
            print("==" * 20)
            continue

        n2 = sum(p.numel() for p in model.docking_attention.parameters())
        print(f"Number of parameters in docking_attention:{n2:,}")
        # print number of parameters for each layer in docking_attention
        for name, param in model.docking_attention.named_parameters():
            print(name, f'{param.numel():,}')
        print("==" * 20)
