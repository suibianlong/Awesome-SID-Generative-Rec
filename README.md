# Awesome Modular Semantic-based Generative Recommendation

> A curated list of resources based on the **Five-Stage Modular Analysis Framework**:
> **Representation** $\rightarrow$ **Tokenization** $\rightarrow$ **Generative Backbone** $\rightarrow$ **Training Paradigm** $\rightarrow$ **Inference**.

## 📖 Taxonomy Overview

This repository organizes Generative Recommendation (GenRec) research not by listing papers chronologically, but by dissecting them into the **modular pipeline**. This perspective reveals how different methods innovate at specific stages of the generation process.

---

## 1. Representation Layer
*Capture the input semantics before discrete quantization.*

| Strategy | Description | Representative Papers |
| :--- | :--- | :--- |
| **Semantic Embedding** | Utilizing PLMs (BERT/ViT) to extract textual or visual features. | **TIGER** (NeurIPS'23), **LETTER** (arXiv'24) |
| **Collaborative / Graph** | Fusing interaction signals (CF) into the semantic space. | **EAGER** (arXiv'23), **LC-Rec** (WWW'24) |
| **Multimodal Unified** | Jointly modeling text, image, and ID features. | **RPG** (KDD'25), **VGA** (ACL'24) |
| **Continuous Interaction** | Directly using raw interaction vectors (for Diffusion models). | **DiffRec** (SIGIR'23), **DDRM** (SIGIR'24) |

---

## 2. Tokenization Layer: Discretization for Generation
*将连续表征离散化为可生成 Token (Codebooks)。*

| Tokenizer Family | Sub-Category | Paper's Tokenization Focus & Details |
| :--- | :--- | :--- |
| **Residual Quantization (RQ)** | **RQ-VAE** | **TIGER** (NeurIPS'23) <br> <img src="./assets/Tokenization-TIGER.png" width="600" /> <br> *利用多层残差量化器将 Item Embedding 编码为固定长度的 Token 序列，首次实现 ID 到 Token 的转换。* |
| |  | **LETTER** (CIKM'24) <br> <img src="./assets/Tokenization-LETTER.png" width="600" /> <br> *提出了可学习的 Tokenizer，通过 **RQ-VAE** (语义正则化)、**对比对齐损失** (协同正则化) 和 **多样性损失** 共同优化代码本，解决了现有代码本缺乏协同信号和分配偏差的问题。* |
| | **R-KMeans** | OneRec  |
| **Product Quantization (PQ)** | **PQ** | RPG |
| | **OPQ** | **RPG** (KDD'25) <br> [RPG并行生成/OPQ示意图] <br> *贡献: 采用了类似 OPQ 的结构来构建 Long Semantic ID，以支持非自回归的并行生成，解决了短 ID 语义容量不足的问题。* |
| **Clustering-based** | **Hierarchical K-Means** | **GenRet** (NeurIPS'22) <br> [GenRet树结构ID生成图] <br> *贡献: 将 Item ID 编码成树路径序列，利用层次 K-Means 聚类构建树结构，将生成问题转化为路径生成。* |
| | **Hierarchical K-Means** | **SEER** (RecSys'23) <br> [SEER ID解释性结构图] <br> *贡献: 采用层次化的离散 ID 结构，利用聚类结果来提供推荐的可解释性，并指导生成过程。* |
| **Hybrid / Textual** | **Raw Text Tokens** | **GPT4Rec** (SIGIR eCom'23) <br> [GPT4Rec Query/Text生成流程图] <br> *贡献: 将用户历史转化为文本 Query，然后生成 Item Title 等描述性文本，完全绕过了 Item ID 语义量化步骤。* |
| | **ID + Text Mixing** | **OneRec** (arXiv'25) <br> [OneRec混合Token输入示意图] <br> *贡献: 将离散的 Item ID Token 和连续的文本 Token 作为 LLM 的输入，实现了统一的检索与排序。* |
| **Learnable / E2E** | **Joint Optimization** | **LC-Rec** (WWW'24) <br> [LC-Rec联合优化框图] <br> *贡献: 提出了可学习的代码本，让量化器在推荐任务中同步优化，以适配生成式骨干。* |
| | **Joint Optimization** | **ETEGRec** (CIKM'24) <br> [ETEGRec端到端架构图] <br> *贡献: 侧重于实现 Tokenizer 与生成模块的端到端可训练性，减少量化误差对推荐性能的影响。* |

---

## 3. Generative Backbone
*The architecture modeling the probability of the token sequence.*

| Architecture | Pros & Cons | Representative Papers |
| :--- | :--- | :--- |
| **Encoder-Decoder (T5/BART)** | Bi-directional context encoding; good for mapping History $\to$ Target. | **TIGER**, **P5**, **VQ-Rec** |
| **Decoder-Only (LLM/GPT)** | Strong reasoning & zero-shot ability; standard for LLM-based approaches. | **OneRec**, **GPT4Rec**, **SGL** |
| **Non-Autoregressive (NAR)** | Parallel generation; significantly faster inference but harder to train. | **RPG** (KDD'25) |
| **Diffusion (Denoising)** | Iterative noise removal; generates continuous vectors, not tokens. | **DiffRec**, **LDiffRec** |

---

## 4. Training Paradigm
*How the system is optimized and aligned.*

| Paradigm | Description | Representative Papers |
| :--- | :--- | :--- |
| **Two-Stage (Quantize $\to$ Train)** | Step 1: Train Codebook (VQ-VAE). Step 2: Train Generator (Seq2Seq). | **TIGER**, **VQ-Rec** |
| **Joint / End-to-End** | Optimizing quantization loss and generation loss simultaneously. | **LC-Rec**, **GeneRec** |
| **Pre-train & Fine-tune** | Standard LLM paradigm: Language Modeling pre-training $\to$ Rec fine-tuning. | **GPT4Rec**, **P5** |
| **Alignment (RLHF/DPO)** | Aligning generation with ranking metrics or user feedback (Reinforcement Learning). | **OneRec** (Preference Alignment), **TallRec** |

---

## 5. Inference & Decoding
*Strategies to generate valid items and rank them efficiently.*

| Strategy | Description | Representative Papers |
| :--- | :--- | :--- |
| **Constrained Beam Search** | Using a Prefix Tree (Trie) to force the generator to output valid Item IDs. | **TIGER**, **GenRet**, **LETTER** |
| **Standard Beam Search** | Generating top-K sequences based on probability (may hallucinate invalid IDs). | **GPT4Rec** |
| **Parallel / Graph Decoding** | Non-autoregressive decoding guided by graph constraints. | **RPG** |
| **Re-ranking / Scoring** | Using the generator to score candidates retrieved by another model. | **LLaRA**, **TALLRec** |

---