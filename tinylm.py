import torch
import tiktoken
from typing import Union, Optional, List
from rotary_embedding_torch import RotaryEmbedding


class TinyLM(torch.nn.Module):
    def __init__(self, 
        embedding_dim: int = 128, 
        activation_dim: int = 512,
        tokenizer_model_name="r50k_base"
    ):
        # FIXME forgot rotary position embeddings
        super().__init__()

        self.tokenizer = tiktoken.get_encoding(tokenizer_model_name)
        self.tokenize = self.tokenizer.encode
        self.decode = self.tokenizer.decode

        self.embedding_dim = embedding_dim
        self.embed = torch.nn.Embedding(self.tokenizer.n_vocab, self.embedding_dim)
        self.rot_emb = RotaryEmbedding(self.embedding_dim)

        self.W_q = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_k = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.W_v = torch.nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.sqrt_dim = torch.sqrt(torch.scalar_tensor(self.embedding_dim))

        self.activation_dim = activation_dim
        self.mlp = torch.nn.Sequential(
           torch.nn.Linear(self.embedding_dim, self.activation_dim),
           torch.nn.ReLU(),
           torch.nn.Linear(self.activation_dim, self.tokenizer.n_vocab)
        )
            

    def forward(self, 
        x, # where x is Batch x Token
        mask: Optional = None # Should be True where attention should be blocked
    ): 
        # FIXME look at lucidrains or flash-attn's implementations to see how to make this more efficient
        if mask is None:
            n_tok = x.shape[-1]
            mask = torch.tril(torch.ones([n_tok, n_tok])).bool()[None, ...] # None for batch dimension broadcasting
            mask = torch.where(mask, 0, -float("Inf"))

        x_embed = self.embed(x) # B x T -> B x T x D
        residual_stream = x_embed

        # FIXME Missing layernorm on hidden states
        q = self.W_q(x_embed)
        k = self.W_k(x_embed)
        v = self.W_v(x_embed)

        q = self.rot_emb.rotate_queries_or_keys(q)
        k = self.rot_emb.rotate_queries_or_keys(k)
        masked_attn = torch.matmul(q, k.transpose(-2, -1)) + mask
        attn_weights = torch.nn.functional.softmax(masked_attn, dim=-1) / self.sqrt_dim
        x_attn = torch.matmul(attn_weights, v)
        # FIXME Missing output projection

        residual_stream = residual_stream + x_attn
        # FIXME Missing pre mlp layernorm
        logits = torch.nn.functional.log_softmax(self.mlp(residual_stream)) # add x for residual stream

        return logits
