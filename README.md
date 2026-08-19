# Transformer from Scratch & Interactive Visualizer


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://transformer-from-scratch-yashmathur.streamlit.app/)

An end-to-end PyTorch implementation of a Transformer language model built entirely from scratch, complete with an interactive **Streamlit** dashboard to visualize internal attention mechanisms and next-token prediction distributions in real time.

👉 **[Experience the Live Interactive Demo Here](https://transformer-from-scratch-yashmathur.streamlit.app/)**

---

## Key Features

* **Custom PyTorch Engine**: Coded core attention mechanisms, positional encodings, multi-head layers, and feed-forward networks from scratch.
* **Hugging Face Hub Integration**: Lightweight repository design—680MB+ model weights (`transformer.pth`) are dynamically fetched at runtime using `huggingface_hub`.
* **Interactive Inspection**:
  * **Side-by-Side Plots**: Real-time rendering of Top-5 next-token probability distributions alongside multi-head attention heatmaps.
  * **Layer & Head Inspection**: Select and inspect attention patterns across all 6 layers and 8 attention heads.
  * **Generation Controls**: Fine-tune sampling behavior with configurable Temperature, Top-K, and Max New Tokens.

---

## Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Total Parameters** | ~174.62 Million |
| **Model Dimension ($d_{\text{model}}$)** | 512 |
| **Key Dimension ($d_k$)** | 64 |
| **Attention Heads ($h$)** | 8 |
| **Layers** | 6 |
| **Tokenizer** | Qwen 2.5 Tokenizer |

---

## Tech Stack

* **Deep Learning Framework**: PyTorch
* **Frontend / Dashboard**: Streamlit
* **Visualization**: Matplotlib, Seaborn
* **Model Hosting**: Hugging Face Hub

![Transformer Architecture](Transformer%20Arch.png)

---

## Local Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yashm006/transformer-from-scratch.git](https://github.com/yashm006/transformer-from-scratch.git)
   cd transformer-from-scratch

