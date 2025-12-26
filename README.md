# Fine‑Tuning DistilBERT with LoRA for Sentence Classification

This repository demonstrates how to fine‑tune **DistilBERT** using **LoRA (Low‑Rank Adaptation)** for a **sentence‑classification task**.  
LoRA injects small trainable matrices into the model, allowing efficient fine‑tuning with **fewer parameters**, **lower memory usage**, and **faster training**.

---

## 🚀 Features

- Fine‑tune DistilBERT using **LoRA adapters**
- Train on any sentence‑classification dataset (binary or multi‑class)
- Hugging Face `transformers` + `peft` integration
- GPU‑friendly and memory‑efficient
- Export LoRA weights separately or merge them into the base model
- Inference script included

---
