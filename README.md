# Automatic Text Summarization with Hybrid Extractive-Abstractive Methods

## Project Overview

This project presents a comprehensive hybrid approach to automatic text summarization that combines the strengths of both extractive and abstractive methods to generate high-quality, coherent, and informative summaries from large textual documents.

### Motivation

In today's information-rich environment, the exponential growth of digital text data makes it increasingly challenging to extract relevant information efficiently. While extractive summarization preserves factual accuracy by selecting important sentences directly from source documents, it often produces summaries lacking coherence and fluency. Conversely, abstractive summarization generates human-like summaries with better readability but risks introducing factual inconsistencies or hallucinations. This project addresses these limitations by developing a unified hybrid framework that leverages the strengths of both approaches.

---

## Table of Contents

1. [Research Problem](#research-problem)
2. [Literature Review](#literature-review)
3. [Proposed Methodology](#proposed-methodology)
4. [System Architecture](#system-architecture)
5. [Implementation Details](#implementation-details)
6. [Datasets](#datasets)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Expected Outcomes](#expected-outcomes)
9. [Challenges and Limitations](#challenges-and-limitations)
10. [Future Work](#future-work)
11. [References](#references)

---

## Research Problem

### Problem Statement

Given a source document \( D \) consisting of multiple sentences, the goal is to generate a concise summary \( S \) that:
- Captures the essential information and key concepts from \( D \)
- Maintains factual accuracy and coherence
- Achieves better readability than pure extractive methods
- Avoids repetition and hallucinations common in pure abstractive methods

### Research Questions

1. How can extractive and abstractive summarization techniques be effectively combined to maximize summary quality?
2. What architectural components are necessary to ensure both factual accuracy and linguistic fluency?
3. How can the hybrid model handle long documents while maintaining computational efficiency?
4. What evaluation metrics best capture the quality improvements of hybrid summarization?

---

## Literature Review

### Extractive Summarization

Extractive summarization selects important sentences or passages directly from the source document without modification. Key techniques include:

- **TextRank**: Graph-based ranking algorithm that treats sentences as vertices and similarity scores as edges
- **TF-IDF**: Statistical approach based on term frequency and inverse document frequency
- **BERT-based Methods**: Using pre-trained transformer models to compute sentence embeddings and select representative sentences
- **KL Divergence**: Optimization approach to minimize divergence between document and summary distributions

**Advantages**: 
- Preserves factual accuracy
- Computationally efficient
- No hallucination risk

**Disadvantages**:
- Limited coherence and fluency
- Redundant information
- Grammatically awkward when assembled

### Abstractive Summarization

Abstractive summarization generates new text that captures the meaning of the source document using natural language generation techniques:

- **Sequence-to-Sequence Models**: Encoder-decoder architectures with attention mechanisms
- **Transformer Models**: BART, T5, PEGASUS designed specifically for summarization tasks
- **Pointer-Generator Networks**: Hybrid mechanism allowing both vocabulary generation and copying from source
- **Coverage Mechanisms**: Tracking attention to prevent repetition

**Advantages**:
- Human-like fluency and coherence
- Concise and informative
- Better handling of paraphrasing

**Disadvantages**:
- Risk of factual inconsistencies
- Hallucination of information
- Computationally expensive
- Requires large training datasets

### Hybrid Approaches

Recent research demonstrates that hybrid approaches achieve superior performance by combining extractive and abstractive methods:

- **Two-Stage Pipeline**: Extract key sentences first, then apply abstractive refinement
- **Joint Training**: Simultaneously optimize for extraction and generation
- **Hierarchical Models**: Process documents at multiple levels (sentence, paragraph, document)

---

## Proposed Methodology

### Overall Approach

The proposed hybrid summarization system consists of three main phases:

#### Phase 1: Extractive Summarization (Content Selection)
Extract salient sentences from the source document that contain the most important information.

#### Phase 2: Abstractive Summarization (Content Refinement)
Refine and rephrase the extracted sentences to generate a fluent, coherent summary.

#### Phase 3: Post-Processing (Quality Enhancement)
Apply coverage mechanisms and repetition reduction techniques to improve final output.

---

## System Architecture

### 1. Data Preprocessing Pipeline

**Input**: Raw text documents

**Steps**:
1. **Text Cleaning**
   - Remove HTML tags, special characters, and unnecessary whitespace
   - Handle contractions and abbreviations
   - Normalize Unicode characters

2. **Sentence Segmentation**
   - Split document into individual sentences using NLTK's Punkt tokenizer
   - Handle edge cases (abbreviations, decimals, etc.)

3. **Tokenization**
   - Word-level tokenization for feature extraction
   - Subword tokenization (WordPiece/BPE) for transformer models

4. **Stopword Removal** (for extractive features only)
   - Remove common words that don't carry significant meaning
   - Use NLTK stopword list with domain-specific customization

5. **Text Normalization**
   - Lowercasing (where appropriate)
   - Lemmatization using spaCy or NLTK

### 2. Extractive Summarization Module

**Model**: BERT-based Extractive Summarizer

**Architecture**:
```
Input Document → BERT Encoder → Sentence Representations → Ranking Layer → Top-K Sentences
```

**Implementation Details**:

1. **Sentence Encoding**
   - Use pre-trained BERT (bert-base-uncased) to generate contextual embeddings
   - Apply interval segment embeddings to distinguish between sentences
   - Extract [CLS] token representations as sentence vectors

2. **Sentence Ranking**
   - **TextRank Scoring**: 
     - Construct similarity graph using cosine similarity between sentence embeddings
     - Apply PageRank algorithm to compute sentence importance scores
   
   - **Position-based Weighting**:
     - Give higher weights to sentences appearing early in document
     - Apply dynamic weighting based on document structure

   - **KL Divergence Optimization**:
     - Compute unigram distribution of document P_D
     - Select sentences that minimize KL(P_D || P_S) where P_S is summary distribution

3. **Sentence Selection**
   - Select top N sentences (N = 3-5 for short documents, 5-10 for long documents)
   - Apply Maximal Marginal Relevance (MMR) to reduce redundancy
   - Maintain sentence order from original document

**Mathematical Formulation**:

**TextRank Score**:
\[
Score(S_i) = (1-d) + d \times \sum_{S_j \in neighbors(S_i)} \frac{sim(S_i, S_j)}{\sum_{S_k \in neighbors(S_j)} sim(S_j, S_k)} \times Score(S_j)
\]

where \( d \) is damping factor (typically 0.85) and \( sim(S_i, S_j) \) is cosine similarity.

**KL Divergence**:
\[
KL(P_D || P_S) = \sum_{w} P_D(w) \log \frac{P_D(w)}{P_S(w)}
\]

### 3. Abstractive Summarization Module

**Model Options**:
- **BART** (facebook/bart-large-cnn): Bidirectional encoder with autoregressive decoder
- **T5** (t5-base or t5-large): Text-to-text transformer framework
- **PEGASUS** (google/pegasus-cnn_dailymail): Pre-trained specifically for summarization

**Selected Model**: BART (Recommended for balanced performance)

**Architecture**:
```
Extracted Sentences → BART Encoder → Contextual Representations → 
BART Decoder (with Attention + Pointer-Generator) → Abstractive Summary
```

**Implementation Details**:

1. **Input Preparation**
   - Concatenate extracted sentences from Phase 1
   - Add special tokens and position encodings
   - Prefix with "summarize:" for T5 or task-specific prompts

2. **Encoder Processing**
   - Bidirectional encoding of input sequences
   - Multi-head self-attention to capture dependencies
   - Layer normalization and residual connections

3. **Decoder Generation**
   - Autoregressive generation with beam search (num_beams=4-5)
   - Cross-attention to encoder states
   - Pointer-generator mechanism for copying capability

4. **Pointer-Generator Network**
   - Generate probability \( P_{gen} \): decides between generating from vocabulary or copying from source
   
   \[
   P_{gen} = \sigma(W_h^T h_t + W_s^T s_t + W_x^T x_t + b)
   \]
   
   - Final word probability:
   
   \[
   P(w) = P_{gen} \times P_{vocab}(w) + (1 - P_{gen}) \times \sum_{i:w_i=w} \alpha_i^t
   \]

5. **Coverage Mechanism**
   - Track cumulative attention: \( c^t = \sum_{t'=0}^{t-1} \alpha^{t'} \)
   - Modify attention calculation to discourage repeated attention
   - Coverage loss: \( L_{cov} = \sum_i \min(\alpha_i^t, c_i^t) \)
   - Total loss: \( L = L_{gen} + \lambda L_{cov} \) where \( \lambda \) is hyperparameter

### 4. Post-Processing Module

1. **Repetition Detection and Removal**
   - Identify repeated n-grams (trigrams and above)
   - Remove redundant sentences using similarity thresholding
   - Apply sentence-level backtracking to prevent duplicate content

2. **Coherence Enhancement**
   - Check pronoun references and resolve dangling anaphora
   - Ensure logical flow between sentences
   - Add discourse connectives if needed

3. **Length Control**
   - Ensure summary meets target length requirements
   - Apply additional pruning or expansion as needed

---

## Implementation Details

### Technology Stack

**Programming Language**: Python 3.8+

**Deep Learning Framework**: PyTorch 2.0+

**Libraries and Tools**:
- **Transformers**: Hugging Face transformers library for pre-trained models
- **NLTK**: Natural Language Toolkit for preprocessing
- **spaCy**: Advanced NLP library for tokenization and lemmatization
- **scikit-learn**: For similarity computations and evaluation metrics
- **NetworkX**: For graph-based algorithms (TextRank)
- **rouge-score**: For ROUGE metric computation
- **nltk.translate.bleu_score**: For BLEU score computation
- **bert-score**: For semantic similarity evaluation

### Model Configuration

**BERT Extractive Module**:
```python
model_name = 'bert-base-uncased'
max_length = 512  # Maximum sequence length
batch_size = 8
num_sentences_extract = 5  # Number of sentences to extract
```

**BART Abstractive Module**:
```python
model_name = 'facebook/bart-large-cnn'
max_input_length = 1024
max_output_length = 150
num_beams = 4
length_penalty = 2.0
early_stopping = True
no_repeat_ngram_size = 3
```

**T5 Alternative Configuration**:
```python
model_name = 't5-base'
prefix = "summarize: "
max_input_length = 512
max_output_length = 150
```

### Training Strategy (if fine-tuning)

1. **Extractive Module Training**:
   - Convert abstractive summaries to extractive labels using greedy ROUGE optimization
   - Binary classification: predict whether each sentence should be included
   - Loss function: Binary Cross-Entropy
   - Optimizer: AdamW with learning rate 2e-5
   - Training epochs: 3-5

2. **Abstractive Module Fine-tuning**:
   - Use extractive summaries as input, abstractive summaries as target
   - Loss function: Cross-entropy for sequence generation
   - Optimizer: AdamW with learning rate 3e-5
   - Training epochs: 3-5
   - Gradient accumulation: 4 steps
   - Mixed precision training (FP16) for efficiency

### Hyperparameter Tuning

Key hyperparameters to tune:
- Number of extracted sentences (3-10)
- Beam search width (3-6)
- Length penalty (1.0-3.0)
- Coverage loss weight λ (0.5-2.0)
- No-repeat n-gram size (2-4)

### Computational Requirements

- **GPU**: NVIDIA GPU with at least 12GB VRAM (e.g., RTX 3060, Tesla T4)
- **RAM**: Minimum 16GB system RAM
- **Storage**: ~5GB for model weights and datasets
- **Inference Time**: 2-5 seconds per document on GPU

---

## Datasets

### 1. CNN/DailyMail Dataset

**Description**: Large-scale dataset of news articles paired with multi-sentence summaries.

**Statistics**:
- Training: 287,113 documents
- Validation: 13,368 documents  
- Test: 11,490 documents
- Average document length: ~760 words
- Average summary length: ~56 words
- Type: News articles (CNN and Daily Mail)

**Access**: Available through Hugging Face Datasets
```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```

### 2. XSum (Extreme Summarization)

**Description**: BBC news articles with single-sentence highly abstractive summaries.

**Statistics**:
- Training: 204,045 documents
- Validation: 11,332 documents
- Test: 11,334 documents
- Average document length: ~431 words
- Average summary length: ~23 words (1 sentence)
- Type: BBC news articles (various topics)

**Access**: Available through Hugging Face Datasets
```python
dataset = load_dataset("xsum")
```

### 3. Scientific Papers Dataset (Optional)

For domain-specific application in academic contexts:
- **ArXiv Dataset**: Computer science papers with abstracts
- **PubMed**: Biomedical literature with abstracts

### Data Preprocessing Workflow

```python
def preprocess_document(document):
    # 1. Text cleaning
    text = clean_text(document)
    
    # 2. Sentence segmentation
    sentences = nltk.sent_tokenize(text)
    
    # 3. Tokenization
    tokens = [word_tokenize(sent) for sent in sentences]
    
    # 4. Remove stopwords (for extractive features)
    filtered_tokens = remove_stopwords(tokens)
    
    # 5. Lemmatization
    lemmatized = lemmatize(filtered_tokens)
    
    return {
        'original_text': text,
        'sentences': sentences,
        'processed_tokens': lemmatized
    }
```

---

## Evaluation Metrics

### 1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Primary metric for summarization evaluation, measuring n-gram overlap between generated and reference summaries.

**ROUGE-1**: Unigram overlap (measures content coverage)
\[
ROUGE-1 = \frac{\sum_{gram \in Reference} Count_{match}(gram)}{\sum_{gram \in Reference} Count(gram)}
\]

**ROUGE-2**: Bigram overlap (measures fluency)

**ROUGE-L**: Longest Common Subsequence (measures coherence and sentence structure)

**Target Scores**:
- ROUGE-1: > 42% (CNN/DailyMail), > 45% (XSum)
- ROUGE-2: > 20% (CNN/DailyMail), > 22% (XSum)
- ROUGE-L: > 38% (CNN/DailyMail), > 42% (XSum)

### 2. BLEU (Bilingual Evaluation Understudy)

Measures n-gram precision with brevity penalty, originally designed for machine translation.

\[
BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
\]

where \( p_n \) is n-gram precision and \( BP \) is brevity penalty.

**Target Score**: > 0.35 for abstractive quality

### 3. METEOR

Considers synonyms, stemming, and paraphrasing in addition to exact matches.

**Advantages**: Better correlation with human judgment than BLEU

**Target Score**: > 0.25

### 4. BERTScore

Semantic similarity metric using contextual embeddings from BERT.

\[
BERTScore = \frac{1}{|x|} \sum_{x_i \in x} \max_{y_j \in y} \text{sim}(x_i, y_j)
\]

**Target F1 Score**: > 0.88

### 5. Human Evaluation Criteria

For qualitative assessment:

1. **Relevance**: Does the summary capture key information? (1-5 scale)
2. **Coherence**: Is the summary logically organized? (1-5 scale)
3. **Fluency**: Is the language natural and grammatical? (1-5 scale)
4. **Non-redundancy**: Does the summary avoid repetition? (1-5 scale)
5. **Factual Consistency**: Is the summary factually accurate? (1-5 scale)

---

## Expected Outcomes

### Quantitative Results

Expected performance improvements over baseline methods:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|-------|---------|---------|---------|------|--------|
| Pure Extractive (BERT) | 40.2 | 17.6 | 36.7 | 0.10 | 0.22 |
| Pure Abstractive (BART) | 44.2 | 21.3 | 40.8 | 0.34 | 0.27 |
| **Proposed Hybrid** | **46.5** | **23.1** | **43.2** | **0.38** | **0.29** |

### Qualitative Improvements

1. **Better Factual Accuracy**: Extractive phase ensures important facts are preserved
2. **Enhanced Fluency**: Abstractive phase generates coherent, readable text
3. **Reduced Redundancy**: Coverage mechanism prevents repetition
4. **Improved Compression**: Higher information density than extractive methods
5. **Balanced Abstraction**: Avoids over-abstraction and hallucination risks

### Use Cases and Applications

1. **News Summarization**: Automatic generation of article summaries for news aggregators
2. **Document Analysis**: Executive summaries for business reports and research papers
3. **Legal Document Processing**: Case law and contract summarization
4. **Medical Literature Review**: Summarizing clinical studies and research findings
5. **Customer Support**: Ticket summarization and FAQ generation
6. **Educational Tools**: Study material summarization for students
7. **Social Media Monitoring**: Summarizing trending topics and discussions

---

## Challenges and Limitations

### Technical Challenges

1. **Computational Complexity**
   - Processing long documents requires significant GPU memory
   - Two-stage pipeline increases inference time
   - Solution: Implement efficient chunking and caching strategies

2. **Information Loss**
   - Extractive phase may miss important nuanced information
   - Risk of losing contextual connections between sentences
   - Solution: Increase number of extracted sentences for longer documents

3. **Coherence in Hybrid Output**
   - Maintaining logical flow between extracted and generated content
   - Handling pronoun references across sentence boundaries
   - Solution: Implement coreference resolution in post-processing

4. **Domain Adaptation**
   - Models trained on news may not generalize to scientific or legal texts
   - Solution: Fine-tune on domain-specific datasets

### Evaluation Challenges

1. **Metric Limitations**
   - ROUGE/BLEU don't capture semantic similarity well
   - High scores don't guarantee summary quality
   - Solution: Use multiple metrics including BERTScore and human evaluation

2. **Reference Summary Variability**
   - Multiple valid summaries possible for same document
   - Single reference may not capture all aspects
   - Solution: Use multiple reference summaries when available

### Limitations

1. **Language Dependency**: Current implementation focuses on English text
2. **Document Length**: Very long documents (>5000 words) may require hierarchical processing
3. **Real-time Constraints**: Two-stage approach increases latency
4. **Training Data Requirements**: Fine-tuning requires large annotated datasets

---

## Future Work

### Short-term Enhancements

1. **Optimization**
   - Implement model quantization (8-bit) for faster inference
   - Apply knowledge distillation to create smaller, efficient models
   - Explore dynamic sentence selection based on document type

2. **Advanced Features**
   - Integrate query-focused summarization for specific information needs
   - Add multi-document summarization capability
   - Implement controllable summarization (length, style, detail level)

3. **Improved Evaluation**
   - Develop factual consistency checkers using NLI models
   - Implement automated coherence scoring
   - Create domain-specific evaluation benchmarks

### Long-term Research Directions

1. **Multilingual Support**
   - Extend to multiple languages using mBART or mT5
   - Cross-lingual summarization (document in one language, summary in another)
   - Low-resource language adaptation

2. **Hierarchical Processing**
   - Develop hierarchical attention mechanisms for very long documents
   - Implement section-aware summarization for structured documents
   - Multi-level abstraction (word → sentence → paragraph → document)

3. **Interactive Summarization**
   - User feedback integration for iterative refinement
   - Personalized summarization based on user preferences
   - Real-time adaptive summarization

4. **Multimodal Summarization**
   - Integrate visual information from figures and tables
   - Video and audio content summarization
   - Cross-modal summary generation

5. **Reinforcement Learning**
   - Apply RL to optimize for specific quality metrics
   - Policy gradient methods for end-to-end training
   - Reward shaping based on human preferences

6. **Explainability**
   - Provide explanations for sentence selection in extractive phase
   - Highlight source sentences contributing to each generated sentence
   - Visualize attention patterns and decision processes

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- Set up development environment and dependencies
- Implement data preprocessing pipeline
- Develop extractive summarization module using BERT
- Initial testing on sample datasets

### Phase 2: Integration (Weeks 5-8)
- Implement abstractive summarization module using BART/T5
- Integrate extractive and abstractive components
- Develop post-processing and quality enhancement
- End-to-end pipeline testing

### Phase 3: Optimization (Weeks 9-12)
- Fine-tune models on target datasets
- Hyperparameter optimization
- Implement coverage mechanisms and repetition handling
- Performance benchmarking

### Phase 4: Evaluation (Weeks 13-15)
- Comprehensive evaluation using automatic metrics
- Conduct human evaluation studies
- Comparative analysis with baseline methods
- Error analysis and iterative improvements

### Phase 5: Documentation (Week 16)
- Write research paper
- Prepare presentation materials
- Create demo application
- Document codebase and create user guide

---

## Web Demo Implementation (This Repository)

The `frontend/` and `backend/` folders implement the hybrid web experience specified above:

- **Backend (`backend/`)** – FastAPI service with the complete Python pipeline: Sentence-BERT + TextRank + handcrafted features for extraction, BART-large with beam search and coverage controls for abstraction, repetition-aware post-processing, and `/evaluate` endpoints for ROUGE/BLEU/METEOR/BERTScore benchmarking.
- **Frontend (`frontend/`)** – React + Vite SPA that captures long-form documents, displays generated summaries alongside the extractive “seed” sentences, and optionally scores model output against human references.

### Running the stack

```bash
# Backend
python -m venv .venv && .venv\Scripts\activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
python backend/main.py  # FastAPI on http://localhost:8000

# Frontend
cd frontend
npm install
npm run dev  # http://localhost:5173
```

Set `VITE_API_BASE_URL` if the API runs on a non-local address.

---

## Code Structure

```
hybrid-summarization/
│
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Preprocessed data
│   └── results/                # Generated summaries
│
├── models/
│   ├── extractive/             # BERT-based extractor
│   ├── abstractive/            # BART/T5 generator
│   └── pretrained/             # Downloaded model weights
│
├── src/
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── extractive_summarizer.py  # Extractive module
│   ├── abstractive_summarizer.py # Abstractive module
│   ├── hybrid_summarizer.py    # Main hybrid pipeline
│   ├── postprocessing.py       # Post-processing utilities
│   └── evaluation.py           # Evaluation metrics
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_extractive_experiments.ipynb
│   ├── 03_abstractive_experiments.ipynb
│   └── 04_hybrid_evaluation.ipynb
│
├── tests/
│   └── test_summarization.py
│
├── config/
│   └── config.yaml             # Configuration parameters
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── main.py                     # Entry point for pipeline
```

---

## Conclusion

This hybrid extractive-abstractive summarization project represents a significant advancement in automatic text summarization by combining the complementary strengths of both approaches. The extractive phase ensures factual accuracy and content coverage, while the abstractive phase enhances fluency and readability. Through careful architecture design, comprehensive evaluation, and iterative refinement, this system aims to produce high-quality summaries that rival human-generated abstracts.

The methodology outlined in this document provides a complete roadmap for implementation, from data preprocessing through model training to final evaluation. By leveraging state-of-the-art transformer models (BERT, BART, T5) and incorporating advanced mechanisms (pointer-generator networks, coverage mechanisms), this hybrid approach addresses the key limitations of existing summarization systems.

The project's modular design allows for flexibility in model selection and easy adaptation to different domains and use cases. With comprehensive evaluation using both automatic metrics (ROUGE, BLEU, METEOR, BERTScore) and human assessment, the system's performance can be rigorously validated and continuously improved.

Future extensions including multilingual support, interactive summarization, and multimodal integration will further enhance the system's capabilities and broaden its applicability to diverse real-world scenarios.

---

## References

### Key Papers

1. See, A., Liu, P. J., & Manning, C. D. (2017). "Get To The Point: Summarization with Pointer-Generator Networks." ACL 2017.

2. Liu, Y., & Lapata, M. (2019). "Text Summarization with Pretrained Encoders." EMNLP-IJCNLP 2019.

3. Lewis, M., et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." ACL 2020.

4. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR 2020.

5. Zhang, J., et al. (2020). "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization." ICML 2020.

6. Divya, S., et al. (2024). "Unified extractive-abstractive summarization: a hybrid approach utilizing BERT and transformer models for enhanced document summarization." PeerJ Computer Science, 10:e2424.

7. Ahmed, R., & Hemanth, D. J. (2025). "Hybrid text summarization: Integrating extractive and abstractive models for enhanced cross-domain summarization." Intelligent Decision Technologies.

8. Yadav, D., Desai, J., & Yadav, A. K. (2022). "Automatic Text Summarization Methods: A Comprehensive Review." arXiv:2204.01849.

### Datasets

9. Hermann, K. M., et al. (2015). "Teaching machines to read and comprehend." NIPS 2015. (CNN/DailyMail)

10. Narayan, S., Cohen, S. B., & Lapata, M. (2018). "Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization." EMNLP 2018. (XSum)

### Evaluation

11. Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." Text Summarization Branches Out Workshop, ACL 2004.

12. Zhang, T., et al. (2020). "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.

---

## Contact and Contribution

**Author**: [Your Name]  
**Institution**: [Your Institution]  
**Course**: Natural Language Processing  
**Semester**: [Current Semester]

For questions, suggestions, or collaboration opportunities, please contact: [Your Email]

---

**Last Updated**: October 2025  
**Version**: 1.0

---

## Appendix A: Installation Guide

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# pip package manager
pip --version
```

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/hybrid-summarization.git
cd hybrid-summarization

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
pip install nltk spacy scikit-learn networkx
pip install rouge-score bert-score
pip install jupyter notebook

# 4. Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 5. Download spaCy model
python -m spacy download en_core_web_sm

# 6. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Quick Start Example

```python
from src.hybrid_summarizer import HybridSummarizer

# Initialize summarizer
summarizer = HybridSummarizer(
    extractive_model='bert-base-uncased',
    abstractive_model='facebook/bart-large-cnn',
    num_sentences=5,
    max_length=150
)

# Summarize document
document = """Your long document text here..."""
summary = summarizer.summarize(document)
print("Summary:", summary)
```

---

## Appendix B: Troubleshooting

### Common Issues

**Issue 1: CUDA Out of Memory**
```
Solution: Reduce batch size, use gradient accumulation, or implement model quantization
```

**Issue 2: Slow Inference**
```
Solution: Use smaller models (t5-small, bart-base), reduce max_length, or use GPU
```

**Issue 3: Poor Summary Quality**
```
Solution: Adjust number of extracted sentences, tune beam search parameters, or fine-tune on domain-specific data
```

---

This comprehensive methodology document provides everything needed to implement, evaluate, and extend your hybrid text summarization project for your NLP course. Good luck with your research!