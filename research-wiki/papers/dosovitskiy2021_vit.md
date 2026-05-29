---
type: paper
node_id: paper:dosovitskiy2021_vit
title: "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"
authors: ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov", "Dirk Weissenborn", "Xiaohua Zhai", "Thomas Unterthiner", "Mostafa Dehghani", "Matthias Minderer", "Georg Heigold", "Sylvain Gelly", "Jakob Uszkoreit", "Neil Houlsby"]
year: 2021
venue: "ICLR"
external_ids:
  arxiv: "2010.11929"
  doi: null
  s2: null
tags: ["vision-transformer", "image-classification", "self-attention", "patch-embedding", "scaling"]
added: 2026-05-29T16:00:00Z
---

# An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale

## One-line thesis

A pure Transformer architecture applied directly to sequences of image patches can achieve state-of-the-art image classification when pre-trained at sufficient scale, outperforming convolutional networks with lower pre-training cost.

## Problem / Gap

The dominant approach in computer vision relied on convolutional architectures (CNNs). While Transformers had become standard in NLP, their application to vision was limited to augmenting CNNs or replacing specific components. Prior attempts at pure self-attention for images either used complex engineering for efficient implementation or underperformed on mid-sized datasets. The question was whether the scalability of Transformers could overcome the lack of CNN-like inductive biases (translation equivariance, locality) in vision.

## Method

ViT splits an image into fixed-size patches (e.g., 16x16), linearly embeds each patch, adds position embeddings, and feeds the resulting sequence to a standard Transformer encoder. A learnable [CLS] token appended to the sequence serves as the image representation for classification. The model uses standard Transformer blocks (multi-head self-attention, MLP, LayerNorm, residual connections). ViT intentionally minimizes vision-specific modifications — the key inductive biases are only at patch extraction and position embedding interpolation during fine-tuning. A hybrid variant uses CNN feature maps as input patches instead of raw pixels.

## Key Results

- Pre-trained on ImageNet-21k (14M images) or JFT-300M (303M images), ViT matches or beats state-of-the-art CNNs on multiple benchmarks
- Best model: 88.55% on ImageNet, 90.72% on ImageNet-ReaL, 94.55% on CIFAR-100, 77.63% on VTAB (19 tasks)
- ViT requires substantially less computational resources to pre-train than comparable CNNs
- Performance scales with dataset size — on mid-sized datasets (ImageNet-1k) ViT underperforms ResNets of comparable size without strong regularization, but on large datasets the gap reverses
- Self-supervised pre-training (masked patch prediction) shows promising initial results

## Assumptions

- Sufficient pre-training data is available (at least 14M images) — the model does not generalize well on small data
- Patch size and position embedding interpolation are sufficient spatial inductive bias; the model assumes 2D structure can be learned from data
- Standard Transformer architecture transfers directly from NLP to vision with minimal modification

## Limitations / Failure Modes

- Requires large-scale pre-training data to outperform CNNs; on small data it underperforms due to lack of built-in inductive biases
- Quadratic self-attention cost in sequence length limits high-resolution image processing
- Position embedding interpolation at higher resolutions during fine-tuning is a heuristic
- Less parameter-efficient than CNNs for small to medium data regimes
- No built-in translation equivariance — the model must learn spatial relationships from scratch

## Reusable Ingredients

- Patch-based image tokenization (16x16 patches as visual "words")
- Pre-trained [CLS] token representation for transfer learning
- Hybrid CNN+Transformer design for leveraging feature maps
- 2D interpolation of position embeddings for variable-resolution fine-tuning
- Scalable proof that large-scale training trumps inductive bias

## Open Questions

- Can ViT benefit from more vision-specific inductive biases without losing scalability?
- How does ViT compare on dense prediction tasks (detection, segmentation) without architectural modifications?
- What is the optimal patch size trade-off between performance and computational cost?

## Claims


## Connections

- **Extended by:** `paper:radford2021_clip` (CLIP — uses ViT as image encoder backbone)
- **Extended by:** `paper:li2023_blip2` (BLIP-2 — uses ViT-based frozen image encoders as visual backbones)

## Relevance to This Project

Foundational architecture for modern vision-language models — both CLIP and BLIP-2 use ViT as a backbone image encoder. Understanding ViT is essential for any VLM research.
