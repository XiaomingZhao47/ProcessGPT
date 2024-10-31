import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import PreTrainedModel, GPT2Config

class GPT(PreTrainedModel):
    # GPT model compatible with Hugging Face Transformers
    config_class = GPT2Config

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.fc = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.fc.weight = self.wte.weight

        # Initialize weights
        self.post_init()

    def forward(self, input_ids, labels=None, attention_mask=None):
        device = input_ids.device
        b_sz, t_sz = input_ids.shape
        pos = torch.arange(0, t_sz, device=device).unsqueeze(0)  # (1, T)

        token_emb = self.wte(input_ids)  # (B, T, E)
        pos_emb = self.wpe(pos)          # (1, T, E)
        x = self.drop(token_emb + pos_emb)  # (B, T, E)

        for block in self.h:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.fc(x)  # (B, T, Vocab)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # This method is used by the generate() function
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {'input_ids': input_ids, 'past_key_values': past}

    @staticmethod
    def _reorder_cache(past, beam_idx):
        # Reorder the past in beam search
        return tuple(layer_past.index_select(0, beam_idx) for layer_past in past)

class DecoderBlock(nn.Module):
    # Decoder block consisting of attention and mlp sub-blocks
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiheadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, attention_mask=attention_mask)
        x = residual + x

        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x
        return x

class MultiheadAttention(nn.Module):
    # Attention sub-block of decoder block
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask to ensure that attention is only applied to previous tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.n_positions, config.n_positions)).view(1, 1, config.n_positions, config.n_positions))

    def _split_heads(self, x):
        # Split the last dimension into (num_heads, head_dim)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (B, num_heads, T, head_dim)

    def _merge_heads(self, x):
        # Merge the num_heads and head_dim dimensions
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.embed_dim,)
        return x.view(*new_shape)

    def forward(self, x, attention_mask=None):
        query, key, value = self.c_attn(x).split(self.embed_dim, dim=2)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Apply causal mask
        attn_weights = attn_weights.masked_fill(self.bias[:, :, :attn_weights.size(-2), :attn_weights.size(-1)] == 0, float('-inf'))

        if attention_mask is not None:
            # Apply attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output

class MLP(nn.Module):
    # Feedforward NN sub-block of decoder block
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Conv1D(nn.Module):
    # huggingface gpt2 implementation uses this as an alternative to nn.Linear
    def __init__(self, out_dim, in_dim):
        super().__init__()
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_dim,)
        # matmul then add bias
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x