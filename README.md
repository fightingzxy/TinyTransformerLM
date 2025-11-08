# Tiny Transformer (Encoder–Decoder) — Mid-term Assignment

From-scratch implementation of a **Transformer Encoder–Decoder** model trained on the **Tiny Shakespeare** dataset (character-level).
Implements:
- Multi-Head Attention (self/cross)
- Position-wise Feed-Forward Network
- Residual Connections + LayerNorm
- Sinusoidal Positional Encoding
- Learning Rate Scheduler & Best Checkpoint Saving

---

## Environment
- Python ≥ 3.9
- PyTorch 2.x (tested on CUDA 11.8, GPU: NVIDIA GeForce RTX 3050 Ti)
- OS: Windows 10 / 11
- Install dependencies:
```bash
pip install -r requirements.txt
