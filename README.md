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
| **Multimodal Unified** | Jointly modeling text, image, and ID features. | **VGA** (ACL'24) |
| **Continuous Interaction** | Directly using raw interaction vectors (for Diffusion models). | **DiffRec** (SIGIR'23), **DDRM** (SIGIR'24) |

| Strategy Family | Sub-Category | Paper's Tokenization & Details |
| :--- | :--- | :--- |
| **Semantic Embedding** | | **TIGER** <br>  <br> *使用语义信息表示item* |
| **Semantic Embedding** | **Multi-source Semantic** | **GREAM** <br>  <br> *使用多源信号（包括物品的标题、官方描述和高质量用户评论），并使用LLM对这些信息进行重写，实现全面、特征丰富的物品描述，最后将重写后的信息使用LLM编码形成嵌入向量* |
| **Collaborative / Graph** | **LLMs** | **LC-Rec** <br>  <br> *使用LLM生成最初的物品表示，他所转换的信息为交互信息，并非单纯的文本信息*  |
| **Collaborative / Graph** | **Dual Codes** | **EAGER** (arXiv'23) <br> [EAGER双流结构图] <br> *同时接受行为及语义信息，两部分分别通过不同的预训练来生成embedding representation，并分别将其传给后面的tokenizer处理。*|

---

## 2. Tokenization Layer: Discretization for Generation
*将连续表征离散化为可生成 Token (Codebooks)。*

| Tokenizer Family | Sub-Category | Paper's Tokenization Focus & Details |
| :--- | :--- | :--- |
| **Residual Quantization (RQ)** | **RQ-VAE** | **TIGER** (NeurIPS'23) <br> <img src="./assets/Tokenization-TIGER.png" width="600" /> <br> *利用多层残差量化器将 Item Embedding 编码为固定长度的 Token 序列，首次实现 ID 到 Token 的转换。* |
| | **RQ-VAE** | **LC-Rec** <br>  <br> *使用RQ-VAE实现从embedding到item ID的转换，本文提出了一种新的解决语义映射冲突的方式，引入uniform distribution constraint，将这个均匀语义映射的任务转换为optimal transmission problem，并通过Sinkhorn-Knopp algorithm解决。* |
| | **RQ-VAE** | **MQL4GRec** <br>  <br> *每个模态（文本、图像）训练一个独立的RQ-VAE翻译器，实现不同模态的信号在输出空间的语义上对齐* |
| | **RQ-VAE** | **LLaDA-Rec** <br>  <br> *提出 Multi-Head VQ-VAE 来生成并行的语义ID。具体做法是：将物品的语义向量分割成M个子向量，每个子向量分别使用一个独立的码本进行量化* |
| |  | **LETTER** (CIKM'24) <br> <img src="./assets/Tokenization-LETTER.png" width="600" /> <br> *提出了可学习的 Tokenizer，通过 **RQ-VAE** (语义正则化)、**对比对齐损失** (协同正则化) 和 **多样性损失** 共同优化代码本，解决了现有代码本缺乏协同信号和分配偏差的问题。* |
| | **R-KMeans** | OneRec  |
| **Product Quantization (PQ)** | **PQ** | RPG |
| | **OPQ** | **RPG** (KDD'25) <br> [RPG并行生成/OPQ示意图] <br> *贡献: 采用了类似 OPQ 的结构来构建 Long Semantic ID，以支持非自回归的并行生成，解决了短 ID 语义容量不足的问题。* |
| **Clustering-based** | **Hierarchical K-Means** | **GenRet** (NeurIPS'22) <br> [GenRet树结构ID生成图] <br> *贡献: 将 Item ID 编码成树路径序列，利用层次 K-Means 聚类构建树结构，将生成问题转化为路径生成。* |
| | **Hierarchical K-Means** | **SEER** (RecSys'23) <br> [SEER ID解释性结构图] <br> *贡献: 采用层次化的离散 ID 结构，利用聚类结果来提供推荐的可解释性，并指导生成过程。* |
| | **Hierarchical K-Means** | **ColaRec** <br> <br> *贡献: 使用层次化K-means对CF信号构建对应的GID* |
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

| Architecture Family | Sub-Category | Paper's Backbone Focus & Details |
| :--- | :--- | :--- |
| **Encoder-Decoder (T5/BART)** | **Transformer** | **TIGER** <br>  <br> *采用标准的 Transformer 编码器-解码器 架构作为序列到序列生成模型* |
| **Encoder-Decoder** | **Two-Stream Generation Architecture**| **EAGER** (arXiv'23) <br> [双流架构图] <br> *使用一个共享编码器和两个解码器，两个解码器分别对应一种序列信息，实现了同时基于两种序列进行预测*  |
| **Only Encoder** | **VQ-Rec**| **VQ-Rec** <br>  <br> *将tokenizer生成的离散code在输入transformer前加了一步Code Embedding Lookup，时模型可以通过微调，改变对应的查找矩阵，以适应不同的下游任务*  |
| **Decoder-Only (LLM/GPT)** | **LLMs**| **LC-Rec** <br>  <br> *模型现在以Sequential Item Prediction为核心任务，使用 Explicit Index-Language Alignment和 Implicit Recommendation-oriented Alignment对原本的LLM进行调整，使其能够通过item ID更准确的预测结果* |
|| **Two-Stream Generation Architecture**| **EAGER** (arXiv'23) <br> [双流架构图] <br> *使用一个共享编码器和两个解码器，两个解码器分别对应一种序列信息，实现了同时基于两种序列进行预测*  |
| **Decoder-Only (LLM/GPT)** | Strong reasoning & zero-shot ability; standard for LLM-based approaches. | **OneRec**, **GPT4Rec**, **SGL** |
| **Non-Autoregressive (NAR)** |  | **RPG** <br>  <br> *1.在序列编码器（Transformer）的最终输出之后，连接了 $m$ 个独立的、并行的预测头（MLP投影头），每个头 $g_j(·)$ 负责预测语义ID中第 $j$ 个位置上的token。2.引入了一个物品级别的聚合表示层。他们先将一个物品的所有token嵌入进行聚合，得到一个单一的物品向量 $v_i$，再将这个物品向量作为序列模型的输入*  |
| **Non-Autoregressive (NAR)** |  | **LLaDA-Rec** <br>  <br> *采用基于双向Transformer的离散扩散模型作为生成模型的核心*  |

---

## 4. Training Paradigm
*How the system is optimized and aligned.*

| Paradigm | Description | Representative Papers |
| :--- | :--- | :--- |
| **Two-Stage (Quantize $\to$ Train)** | Step 1: Train Codebook (VQ-VAE). Step 2: Train Generator (Seq2Seq). | **TIGER**, **VQ-Rec** |
| **Joint / End-to-End** | Optimizing quantization loss and generation loss simultaneously. | **LC-Rec**, **GeneRec** |
| **Pre-train & Fine-tune** | Standard LLM paradigm: Language Modeling pre-training $\to$ Rec fine-tuning. | **GPT4Rec**, **P5** |
| **Alignment (RLHF/DPO)** | Aligning generation with ranking metrics or user feedback (Reinforcement Learning). | **OneRec** (Preference Alignment), **TallRec** |

| Paradigm Family | Sub-Category | Paper's Paradigm Focus & Details |
| :--- | :--- | :--- |
| **Two-Stage (Quantize $\to$ Train)** | | **TIGER** <br> <br> *Step 1: Train Codebook (VQ-VAE). Step 2: Train Generator (Seq2Seq).* |
| **Two-Stage (Quantize $\to$ Train)** | **Multi-Task Enhancement** | **EAGER** (arXiv'23) <br> [双流架构图] <br> *添加了一个Global Ctracive Task和一个Semantic-guided transformer Task，前一个加强了模型对tokens全局的判别能力，后者使由双流模型的两个流之间之间产生信息交换，这两者都是辅助训练目标* |
| **Two-Stage (Quantize $\to$ Train)** | **Multi-Task Enhancement** | **ColaRec**  <br> <br> *引入 Item-Item Indexing和contrastive loss，使模型能够理解从item到GID的映射及拉近相似GID在语义嵌入上的距离，实现语义和CF信号的对其机制。* |
| **Two-Stage (Quantize $\to$ Train)** | **Mix of Generative and Dense Retrieval** | **LIGER**  <br> <br> *使用余弦相似度损失与the next-token prediction loss形成双目标损失，实现稠密检索与生成式检索结合的效果* |
| **Two-Stage (Quantize $\to$ Train)** | **MTP** | **RPG** <br> <br> *引入并适应了“多token预测”目标。该目标让模型同时预测语义ID的所有token，从而打破了token间的顺序依赖，实现并行解码。* |
| **Two-Stage (Quantize $\to$ Train)** | **双层次掩码训练** | **LLaDA-Rec** <br> <br> *使用User-History Level Masking，实现物品间的序列依赖关系的学习，使用Next-Item Level Masking，实现型物品内的语义结构的学习* |
| **Joint / End-to-End** | **alignment tuning** | **LC-Rec** <br>  <br> *模型现在以Sequential Item Prediction为核心任务，使用 Explicit Index-Language Alignment和 Implicit Recommendation-oriented Alignment对原本的LLM进行调整，实现llm与item ID的对齐* |
| **Joint / End-to-End** | **Quantize与Train协同** | **ETEGRec** <br>  <br> *将item tokenization和generative recommendation在一个框架中联合优化，并提出Sequence-Item Alignment和Preference-Semantic Alignment对齐策略，实现序列到物品分布以及偏好到语义的对齐；同时提出交替优化策略，解决端到端训练的不稳定性* |
| **Joint / End-to-End** | **以end-to-end为骨架，嵌入alignment** | **GREAM** <br>  <br> *创建多层次对齐任务，包括顺序推荐任务，语义重建任务及用户偏好任务，实现在LLM的表示空间中，建立一个同时承载了语言语义和协作语义的混合表示；在传统推理链上加入去噪序列重写，实现模型主动识别并过滤掉用户历史中的噪声交互；使用合成CoT数据，反向推理生成高质量数据集，将SFT与课程学习调度相融合；提出SRPO 算法（包括残差敏感的可验证奖励，奖励校准的优势估计和方差引导的动态采样），解决应用RLHF会因奖励稀疏导致训练不稳定、难以收敛* |
| **Pre-train & Fine-tune** | | **VQ-Rec**  <br>  <br> *在传统的预训练-微调范式的基础上，在预训练时，使用对比学习，保证模型在多领域的适应性，在微调阶段提出permutation-based alignment完成对code embedding table对于不同下游任务的调整* |
| **Pre-train & Fine-tune** | **Cross-modal** | **MQL4GRec**  <br>  <br> *引入Asymmetric Item Generation 和 Quantitative Language Alignment 实现模态间的知识迁移与对齐* |
| **Alignment (RLHF/DPO)** | **Alignment+fine-tune** | **OneRec**  <br>  <br> *使用荐系统的数据（物品的语义ID序列）对其进行监督微调，在使用GRPO来对齐和优化模型的生成偏好（具体方法为全流程SID-文本对齐和排序感知的奖励函数）* |

---

## 5. Inference & Decoding
*Strategies to generate valid items and rank them efficiently.*

| Strategy | Description | Representative Papers |
| :--- | :--- | :--- |
| **Constrained Beam Search** | Using a Prefix Tree (Trie) to force the generator to output valid Item IDs. | **TIGER**, **GenRet**, **LETTER** |
| **Standard Beam Search** | Generating top-K sequences based on probability (may hallucinate invalid IDs). | **GPT4Rec** |
| **Parallel / Graph Decoding** | Non-autoregressive decoding guided by graph constraints. | **RPG** |
| **Re-ranking / Scoring** | Using the generator to score candidates retrieved by another model. | **LLaRA**, **TALLRec** |

| Family | Sub-Category | Details |
| :--- | :--- | :--- |
| **Constrained Beam Search** | **beam serch + Temperature-based sampling** | **TIGER** <br> <br> *使用 Beam Search 进行自回归解码以生成语义ID，并引入了温度采样来控制推荐多样性* |
| **Constrained Beam Search** | **Conffdence-based Ranking** | **EAGER** (arXiv'23) <br> [双流架构图] <br> *在双流模型的每个流中使用beam search，分别选出各自的top-k，然后对这2k的预测结果中，计算他们的log probabilities，值越小代表可能性越大，对这2k个结果的分数进行排序，取top-k为结果* |
| | **beam serch** | **LC-Rec**  <br>  <br> *使用beam search替换LLM原本的候选集inference方法，并且利用KV buffer加速这一过程，并且在生成过程中过滤掉非法索引组合。* |
| | **适用于离散扩散** | **LC-Rec**  <br>  <br> *包含三步：1.自适应生成顺序：在每一步，选择模型置信度最高（而非固定位置）的token进行生成。2.束搜索集成：在选定的位置上进行束扩展和剪枝，以生成top-k推荐结果。3.迭代精炼：生成某些位置后，对剩余低置信度的位置进行重新掩码，在后续步骤中利用新的上下文信息重新预测。* |
| **Standard Beam Search** | **Multi - mode** | **GREAM** <br>  <br> *有两种推荐模式，一种基于CF signals,另一种先基于逻辑生成再基于CF但是两步筛选都基于beam search* |
| **Parallel / Graph Decoding** | | **RPG**<br>  <br> *为所有有效的语义ID构建一个图结构，并通过迭代的图传播来引导解码过程，实现了在巨大的候选空间中找到高分的、有效的语义ID，而不需要枚举所有物品，本质是先通过之前构建的解码图找到高分候选的相似语义ID，再计算score得到新的高分候选，由此迭代q次，最终找到最优* |
| **Re-ranking / Scoring** | **beam serch+scoring+ranking** |  **LIGER** <br> <br> *inference分为两步，第一步使用beam search得到一个初始候选列表，在使用一个外部打分器（使用稠密检索的相似度计算） 来对生成模型得到的候选列表进行重新排名*  |
| **Re-ranking / Scoring** | **scoring+ranking** |  **MQL4GRec** <br> <br> *模型同时会输出基于图像与文本的偏好推荐列表，将物品在不同列表的分数相加（如果该物品只在一个列表则只算这一个列表中的得分）作为最终得分，再对两个列表的内容进行重排，取top-k*  |

---
