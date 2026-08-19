import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from transformers import AutoTokenizer      # Only for Tokenization


class Tokenisation:

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        print(f"Using {model_name} Tokeniser")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def encode(self, text):
        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

    def encode_batch(self, texts):
        return self.tokenizer(
            texts,
            truncation=False,
            return_attention_mask=False,
            add_special_tokens=False
        )

    
class InputPositionalEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int, max_seq_len:int=512):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.position_embedding = nn.Embedding(num_embeddings=max_seq_len, embedding_dim=d_model)

    def forward(self, x:torch.Tensor):
        # x shape: (batch_size, sequence_length)
        token_emb = self.token_embedding(x)

        # Create position IDs: 0, 1, 2, ..., sequence_length - 1
        positions = torch.arange(
            x.size(1),
            device=x.device
        )

        # position_emb shape: (sequence_length, d_model)
        position_emb = self.position_embedding(positions)

        return token_emb + position_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, h:int, d_model:int, d_k:int):
        super().__init__()
        self.h = h
        assert h * d_k == d_model, f"Embedding dimension d_model ({d_model}) must equal num_heads ({h}) * head_dim ({d_k})"

        self.d_k = d_k 
        self.d_model = d_model

        self.W_Q = nn.Parameter(
            torch.empty(h, d_model, d_k)
        )

        self.W_K = nn.Parameter(
            torch.empty(h, d_model, d_k)
        )

        self.W_V = nn.Parameter(
            torch.empty(h, d_model, d_k)
        )

        self.W_O = nn.Parameter(
            torch.empty(d_model, d_model)
        )

        # Proper initialization
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

    def forward(self, X:torch.Tensor):
        outputs = []
        attentions = []

        seq_len = X.shape[1]

        mask = torch.triu(torch.ones(seq_len, seq_len, device=X.device), diagonal=1).bool()

        for head in range(self.h):
            Q = X @ self.W_Q[head]
            K = X @ self.W_K[head]
            V = X @ self.W_V[head]


            scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)    # (Q.K)/sqrt(d_k)

            scores = scores.masked_fill(mask=mask, value=float('-inf'))

            attention = torch.softmax(scores, dim=-1)
            attentions.append(attention)

            head_output = attention @ V     # head_i = attention * V

            outputs.append(head_output)

        outputs = torch.cat(outputs, dim=-1)
        outputs = outputs @ self.W_O

        return outputs, attentions

    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=1024):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, X:torch.Tensor):
        return self.linear2(torch.nn.functional.gelu(self.linear1(X)))

    
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.d_model = d_model

    def forward(self, X: torch.Tensor):
        mean = X.mean(dim=-1, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=-1, keepdim=True)

        normalized = (X - mean) / torch.sqrt(var + self.eps)

        return self.alpha * normalized + self.beta

    
class ResidualConnection(nn.Module):        # Y = LayerNorm(X + Attention(X))   ----> ADD & NORM
    def __init__(self, d_model:int):
        super().__init__()
        self.norm = LayerNormalization(d_model)

    def forward(self, X:torch.Tensor, sublayer:torch.Tensor):
        return self.norm(X + sublayer)

    
class SingleTransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_k, h: int):
        super().__init__()

        self.MHA = MultiHeadAttention(h, d_model, d_k)
        self.FFN = FeedForwardNetwork(d_model, d_ff=2048)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, X):
        # Pre-LN Attention
        attention_input = self.norm1(X)
        attention_output, attentions = self.MHA(attention_input)        # attention_output = Attention@V, attentions = softmax(QK.T)
        X = X + attention_output

        # Pre-LN FFN
        ffn_input = self.norm2(X)
        ffn_output = self.FFN(ffn_input)
        X = X + ffn_output

        return X, attentions


        
class OutputLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.norm = LayerNormalization(d_model)
        self.linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, X:torch.Tensor):
        X = self.norm(X)
        logits = self.linear(X)
        return logits



class Transformer(nn.Module):
    def __init__(self, d_model:int, d_k, h:int=8):
        super().__init__()

        # Tokeniser
        self.tokeniser = Tokenisation()
        self.vocab_size = self.tokeniser.vocab_size

        # Embedding
        self.InputPositionalEmbedding = InputPositionalEmbeddings(self.vocab_size, d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            SingleTransformerBlock(d_model, d_k, h)
            for _ in range(6)
        ])

        # Output Layer
        self.output = OutputLayer(d_model, self.vocab_size)

        # Parameter Count & Device Verification
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gpu_params = sum(p.numel() for p in self.parameters() if p.is_cuda and p.requires_grad)

        print("=" * 45)
        print(f"Total Trainable Parameters : {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Parameters Stored on GPU    : {gpu_params:,} ({gpu_params/1e6:.2f}M)")
        print("=" * 45)

    def forward(self, X: torch.Tensor, return_attentions: bool = False):
        X = self.InputPositionalEmbedding(X)

        all_attentions = [] if return_attentions else None

        for block in self.blocks:
            X, attentions = block(X)
            if return_attentions:
                all_attentions.append(attentions)

        return self.output(X), all_attentions


    def tokenise(self, sentences, max_length=20):
        tokens = self.tokeniser.encode_batch(
            sentences,
        )
        return tokens["input_ids"]

    @torch.no_grad()
    def generate(model, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
        device = next(model.parameters()).device

        # Tokenise prompt
        tokens = model.tokenise([prompt])[0]
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        model.eval()

        attention_scores  = []

        for _ in range(max_new_tokens):

            # Keep only the most recent seq_len tokens if necessary
            x_input = x[:, -512:]

            # Forward pass
            logits, scores = model(x_input, return_attentions=True)
            attention_scores.append(scores)

            # We only care about the LAST token position
            next_token_logits = logits[:, -1, :]

            # Temperature
            next_token_logits = next_token_logits / temperature

            # Top-k sampling
            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                
                filtered_logits = torch.full_like(
                    next_token_logits,
                    float("-inf")
                )

                filtered_logits.scatter_(1, indices, values)

                next_token_logits = filtered_logits

            # Convert logits → probabilities
            probabilities = torch.softmax(next_token_logits, dim=-1)

            # Sample one token
            next_token = torch.multinomial(probabilities, num_samples=1)

            # Append it
            x = torch.cat([x, next_token], dim=1)

        # Convert token IDs back to text
        generated_tokens = x[0].tolist()

        text = model.tokeniser.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return text, attention_scores

