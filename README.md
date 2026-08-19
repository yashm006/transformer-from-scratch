```text
                  RAW TEXT
                     │
                     ▼
                 Tokenizer
                     │
                     ▼
                Token IDs
                     │
                     ▼
              Token Embedding
                     │
                     ├──────────────┐
                     │              │
                     ▼              ▼
               Positional      (added to)
                Encoding
                     │
                     ▼
              Input Embeddings
                     │
                     ▼
        ┌────────────────────────────┐
        │                            │
        │     TRANSFORMER BLOCK 1    │
        │                            │
        └────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │     TRANSFORMER BLOCK 2    │
        └────────────────────────────┘
                     │
                    ...
                     │
                     ▼
        ┌────────────────────────────┐
        │     TRANSFORMER BLOCK N    │
        └────────────────────────────┘
                     │
                     ▼
                Final LayerNorm
                     │
                     ▼
              Linear Projection
                     │
                     ▼
                  Logits
                     │
                     ▼
                 Softmax
                     │
                     ▼
          Next-token probabilities
                     │
                     ▼
              Predicted token
```

<img src="Transformer Arch.png">
