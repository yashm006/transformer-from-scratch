import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import urllib.request
from huggingface_hub import hf_hub_download

from model import Transformer


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "transformer.pth"
MODEL_URL = "https://huggingface.co/Yashmathur/Transformer-from-scratch"

D_MODEL = 512
D_K = 64
NUM_HEADS = 8

MAX_CONTEXT = 512


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="My Transformer Visualizer", layout="wide")

st.title("My Transformer — Attention & Prediction Visualizer")

st.write("Visualize text generation, next-token prediction probabilities, and attention matrices side-by-side.")


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.write(f"**Device:** `{device}`")

if device.type == "cuda":
    st.sidebar.write(f"**GPU:** `{torch.cuda.get_device_name(0)}`")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    # Download weights directly from Hugging Face Hub
    weights_path = hf_hub_download(
        repo_id="Yashmathur/Transformer-from-scratch", 
        filename="transformer.pth"
    )

    model = Transformer(d_model=D_MODEL, d_k=D_K, h=NUM_HEADS)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


model = load_model()


# ============================================================
# TOKENIZATION
# ============================================================

def get_tokens(text):
    token_ids = model.tokeniser.encode(text)
    tokens = [model.tokeniser.tokenizer.decode([token_id]) for token_id in token_ids]
    return token_ids, tokens


# ============================================================
# GET TOP-5 PREDICTIONS FOR A PARTICULAR STEP
# ============================================================

@torch.no_grad()
def get_top_predictions(token_ids):
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x, return_attentions=False)
    next_token_logits = logits[:, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, 5, dim=-1)

    results = []
    for token_id, probability in zip(top_indices[0].tolist(), top_probs[0].tolist()):
        token = model.tokeniser.tokenizer.decode([token_id])
        results.append({"token": token, "probability": probability})

    return results


# ============================================================
# SIDEBAR (NUMBER INPUTS)
# ============================================================

st.sidebar.header("Generation Options")

temperature = st.sidebar.number_input("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
max_new_tokens = st.sidebar.number_input("Max new tokens", min_value=1, max_value=100, value=10, step=1)
top_k = st.sidebar.number_input("Top-K", min_value=1, max_value=100, value=50, step=1)


# ============================================================
# PROMPT
# ============================================================

prompt = st.text_area("Enter your prompt", value="Once upon a time", height=100)


# ============================================================
# GENERATE
# ============================================================

if st.button("Generate", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Running Transformer..."):
            generated_text, attention_scores = model.generate(
                prompt, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                top_k=top_k
            )

        st.session_state["generated_text"] = generated_text
        st.session_state["attention_scores"] = attention_scores
        st.session_state["prompt"] = prompt


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "attention_scores" in st.session_state:

    generated_text = st.session_state["generated_text"]
    attention_scores = st.session_state["attention_scores"]

    st.header("Generated Text")
    st.info(generated_text)
    st.divider()

    # Get final sequence details automatically
    all_token_ids, all_tokens = get_tokens(generated_text)
    final_attention = attention_scores[-1]
    sequence_length = final_attention[0][0].shape[-1]

    step_token_ids = all_token_ids[:sequence_length]
    step_tokens = [model.tokeniser.tokenizer.decode([token_id]) for token_id in step_token_ids]

    # Controls for Layer and Head selection
    st.header("Model Analysis")
    col1, col2 = st.columns(2)
    with col1:
        layer_number = st.selectbox("Layer", options=list(range(1, 7)), index=5)
    with col2:
        head_number = st.selectbox("Attention Head", options=list(range(1, NUM_HEADS + 1)), index=0)

    # Fetch selected attention matrix
    selected_layer = final_attention[layer_number - 1]
    selected_attention = selected_layer[head_number - 1]
    attention_matrix = selected_attention[0].detach().cpu().numpy()

    # Get top 5 predictions for the current step
    top_predictions = get_top_predictions(step_token_ids)
    
    # Prepare data for horizontal bar graph (reversed for top-down display)
    pred_tokens = [p["token"].replace("\n", "\\n") for p in reversed(top_predictions)]
    pred_probs = [p["probability"] * 100 for p in reversed(top_predictions)]

    # ========================================================
    # SINGLE FIGURE WITH SIDE-BY-SIDE AXES
    # ========================================================

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1.4]}
    )

    # Left Plot: Horizontal Bar Graph for Predictions
    bars = ax_bar.barh(pred_tokens, pred_probs, color="green")
    ax_bar.set_xlabel("Probability (%)", weight='bold', fontsize=14)
    ax_bar.set_title("Top-5 Next Token Predictions", fontsize=18)
    ax_bar.set_xlim(0, 105)

    for bar in bars:
        width = bar.get_width()
        ax_bar.text(
            width + 1.5, 
            bar.get_y() + bar.get_height() / 2, 
            f"{width:.1f}%", 
            va="center", 
            ha="left", 
            fontsize=9
        )

    # Right Plot: Attention Heatmap
    sns.heatmap(
        attention_matrix,
        xticklabels=step_tokens,
        yticklabels=step_tokens,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax_heat,
        cbar_kws={'label': 'Attention Weight'}
    )
    ax_heat.set_xlabel("Token Being Attended To", weight='bold', fontsize=14)
    ax_heat.set_ylabel("Current Token", weight='bold', fontsize=14)
    ax_heat.set_title(f"Attention Matrix (Layer {layer_number}, Head {head_number})", fontsize=18)
    ax_heat.tick_params(axis="x", rotation=45)
    ax_heat.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)