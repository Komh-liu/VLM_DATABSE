---
type: paper
node_id: paper:radford2021_clip
title: "Learning Transferable Visual Models From Natural Language Supervision"
authors: ["Alec Radford", "Jong Wook Kim", "Chris Hallacy", "Aditya Ramesh", "Gabriel Goh", "Sandhini Agarwal", "Girish Sastry", "Amanda Askell", "Pamela Mishkin", "Jack Clark", "Gretchen Krueger", "Ilya Sutskever"]
year: 2021
venue: "arXiv"
external_ids:
  arxiv: "2103.00020"
  doi: null
  s2: null
tags: ["contrastive-learning", "vision-language", "zero-shot", "multi-modal", "image-text"]
added: 2026-05-29T16:00:00Z
---

# Learning Transferable Visual Models From Natural Language Supervision

## One-line thesis

Pre-training an image encoder and text encoder jointly with a contrastive objective on 400 million (image, text) pairs yields transferable visual representations that enable zero-shot classification on diverse downstream tasks without task-specific training.

## Problem / Gap

Computer vision systems were traditionally trained to predict fixed predetermined object categories, limiting their generality. Learning from raw text about images was a promising direction but had demonstrated much lower performance than supervised approaches. Prior attempts at natural language supervision for vision either used small datasets (MS-COCO, Visual Genome, YFCC100M) or complex generative objectives; scaling to web-scale data was underexplored.

## Method

CLIP (Contrastive Language-Image Pre-training) trains an image encoder and a text encoder jointly to predict correct pairings in a batch of (image, text) examples. The contrastive objective maximizes cosine similarity between matched image-text pairs while minimizing it for mismatched pairs. Both encoders can be various architectures (ViT or ResNet for images; Transformer for text). After pre-training, natural language is used to reference visual concepts, enabling zero-shot transfer — the text encoder synthesizes a classifier by embedding class names/descriptions, and prediction is done by selecting the highest similarity score. Training uses a dataset of 400M (image, text) pairs called WIT (WebImageText).

## Key Results

- Zero-shot CLIP matches ResNet-50 accuracy on ImageNet without using any of its 1.28M training examples
- Competitive with fully supervised baselines on over 30 datasets spanning OCR, action recognition, geo-localization, fine-grained classification
- Zero-shot CLIP is much more robust than equivalent-accuracy supervised ImageNet models
- Transfer performance is a smoothly predictable function of compute
- Linear-probe CLIP outperforms the best publicly available ImageNet model while being more computationally efficient

## Assumptions

- Contrastive learning is sufficient for vision-language alignment (more efficient than generative/predictive approaches — 4x more efficient than bag-of-words prediction, 12x more than transformer language model)
- Web-scale data (400M image-text pairs) covers a sufficiently broad set of visual concepts
- Natural language is an effective supervision signal for learning visual representations

## Limitations / Failure Modes

- Zero-shot performance on fine-grained or specialized tasks (e.g., medical imaging, satellite imagery) can be poor if the concepts are not well-represented in WIT
- The contrastive objective may miss finer-grained visual understanding that generative objectives capture
- Requires careful engineering to scale contrastive learning to hundreds of millions of examples
- Text encoder quality depends on the diversity and quality of captions in the training data
- Can exhibit social biases present in web-crawled data

## Reusable Ingredients

- Contrastive image-text pre-training paradigm (now standard in VLM research)
- Dual-encoder architecture for separate image and text representation
- Zero-shot transfer via natural language class descriptors
- Text-prompt engineering for downstream task adaptation
- WIT dataset construction methodology (500K query-based balanced collection)

## Open Questions

- How much further can CLIP-style models scale with more data and compute?
- Can generative objectives complement contrastive learning for better fine-grained understanding?
- How to improve zero-shot performance on out-of-distribution or rare visual concepts?

## Claims


## Connections

- **Extends:** `paper:dosovitskiy2021_vit` (ViT — uses ViT as one of its image encoder backbones)
- **Extended by:** `paper:li2023_blip2` (BLIP-2 — uses frozen CLIP image encoders and builds on contrastive VLP)

## Relevance to This Project

CLIP is the foundational vision-language model that enables zero-shot transfer. BLIP-2 builds directly on CLIP-style pre-trained image encoders. CLIP's contrastive objective is a key component in modern VLMs.
