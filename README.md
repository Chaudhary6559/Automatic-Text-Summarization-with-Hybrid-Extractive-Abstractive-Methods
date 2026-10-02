# Automatic Text Summarization with Hybrid Extractive-Abstractive Methods

## Project Overview

This project presents a comprehensive hybrid approach to automatic text summarization that combines the strengths of both extractive and abstractive methods to generate high-quality, coherent, and informative summaries. The system leverages state-of-the-art pre-trained transformer models (BERT, BART, T5) integrated with traditional NLP techniques.

### Motivation

In today's information-rich environment, the exponential growth of digital text data makes it increasingly challenging to extract relevant information efficiently. While extractive summarization preserves factual accuracy but suffers from poor coherence, abstractive summarization offers better readability but risks hallucinations. This hybrid approach combines the best of both worlds.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Research Problem](#research-problem)
3. [Proposed Methodology](#proposed-methodology)
4. [System Architecture](#system-architecture)
5. [Implementation Details](#implementation-details)
6. [Datasets](#datasets)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Installation and Setup](#installation-and-setup)
9. [Running the Application](#running-the-application)
10. [Evaluation Results](#evaluation-results)
11. [Challenges and Limitations](#challenges-and-limitations)
12. [Future Work](#future-work)
13. [References](#references)

---

## Project Structure

```
Automatic-Text-Summarization-with-Hybrid-Extractive-Abstractive-Methods/
│
├── README.md                          # Project documentation
├── QUICKSTART.md                      # Quick start guide
├── FILE_UPLOAD_FEATURES.md           # File upload capabilities
├── FRONTEND_SETUP.md                 # Frontend setup instructions
├── run_backend.bat                   # Windows batch script to run backend
│
├── backend/                           # FastAPI backend service
│   ├── main.py                       # Entry point
│   ├── app.py                        # FastAPI application
│   ├── requirements.txt              # Python dependencies
│   ├── OCR_SETUP.md                  # OCR configuration guide
│   ├── config/                       # Configuration files
│   ├── src/                          # Source code modules
│   │   ├── preprocessing.py         # Text cleaning and tokenization
│   │   ├── extractive_summarizer.py # BERT-based extraction
│   │   ├── abstractive_summarizer.py# BART/T5 generation
│   │   ├── hybrid_summarizer.py     # Main hybrid pipeline
│   │   ├── postprocessing.py        # Quality enhancement
│   │   └── evaluation.py            # Evaluation metrics
│   └── tests/                        # Unit and integration tests
│
├── frontend/                          # React/Vite web interface
│   ├── index.html                    # Main HTML entry point
│   ├── standalone.html               # Standalone demo (self-contained)
│   ├── package.json                  # Node dependencies
│   ├── package-lock.json             # Dependency lock file
│   ├── vite.config.js                # Vite configuration
│   └── src/                          # React components and logic
│       ├── App.jsx                   # Main application component
│       ├── components/               # UI components
│       ├── styles/                   # CSS styling
│       └── utils/                    # Helper functions
│
└── docs/                              # Documentation
    ├── architecture.md               # System architecture details
    ├── deployment.md                 # Deployment instructions
    ├── evaluation.md                 # Evaluation methodology
    ├── assignment1.txt               # Course assignment 1
    └── assignment2.txt               # Course assignment 2
```

---

## Research Problem

### Problem Statement

Given a source document consisting of multiple sentences, the goal is to generate a concise summary that:
- Captures the essential information and key concepts
- Maintains factual accuracy and coherence
- Achieves better readability than pure extractive methods
- Avoids repetition and hallucinations common in pure abstractive methods

### Research Questions

1. How can extractive and abstractive summarization techniques be effectively combined to maximize summary quality?
2. What architectural components are necessary to ensure both factual accuracy and linguistic fluency?
3. How can the hybrid model handle long documents while maintaining computational efficiency?
4. What evaluation metrics best capture the quality improvements of hybrid summarization?

---

## Proposed Methodology

### Overall Approach

The proposed hybrid summarization system consists of three main phases:

#### Phase 1: Extractive Summarization (Content Selection)
Extract salient sentences from the source document that contain the most important information using BERT-based embeddings and ranking algorithms.

#### Phase 2: Abstractive Summarization (Content Refinement)
Refine and rephrase the extracted sentences using transformer models (BART/T5) to generate a fluent, coherent summary.

#### Phase 3: Post-Processing (Quality Enhancement)
Apply coverage mechanisms and repetition reduction techniques to improve final output quality.

---

## System Architecture

### Overview

```
Input Document
    ↓
[Data Preprocessing Pipeline]
    ↓
[Extractive Module - BERT + TextRank]
    ↓
[Extracted Key Sentences]
    ↓
[Abstractive Module - BART/T5]
    ↓
[Generated Summary]
    ↓
[Post-Processing Module]
    ↓
Final Summary
```

### 1. Data Preprocessing Pipeline

**Steps**:
1. **Text Cleaning** - Remove HTML tags, special characters, normalize whitespace
2. **Sentence Segmentation** - Split into sentences using NLTK Punkt tokenizer
3. **Tokenization** - Word-level and subword tokenization
4. **Stopword Removal** - Remove common words (for extractive features)
5. **Text Normalization** - Lemmatization and lowercasing

### 2. Extractive Summarization Module

**Model**: BERT-based Extractive Summarizer

**Architecture**:
- Input: Document → BERT Encoder → Sentence Representations → Ranking Layer → Top-K Sentences
- Uses BERT (bert-base-uncased) for contextual embeddings
- Implements TextRank scoring with cosine similarity graphs
- Applies PageRank algorithm for importance computation
- Reduces redundancy using Maximal Marginal Relevance (MMR)

**Key Features**:
- Position-based weighting (earlier sentences weighted higher)
- KL Divergence optimization for content selection
- Maintains original sentence order

### 3. Abstractive Summarization Module

**Model Options**:
- BART (facebook/bart-large-cnn) - Recommended
- T5 (t5-base or t5-large)
- PEGASUS (google/pegasus-cnn_dailymail)

**Architecture**:
- Extracted Sentences → BART Encoder → Contextual Representations
- BART Decoder with Attention + Pointer-Generator → Abstractive Summary

**Advanced Features**:
- Beam search (num_beams=4-5)
- Pointer-Generator Network (copy vs. generate)
- Coverage mechanism (prevent repetition)
- Length penalty and no-repeat n-gram control

### 4. Post-Processing Module

1. **Repetition Detection and Removal** - Identify and remove redundant n-grams
2. **Coherence Enhancement** - Resolve pronouns and anaphora
3. **Length Control** - Ensure target length requirements

---

## Implementation Details

### Technology Stack

**Programming Language**: Python 3.8+

**Deep Learning Framework**: PyTorch 2.0+

**Key Libraries**:
- **Transformers**: Hugging Face transformers (BERT, BART, T5)
- **NLTK**: Text preprocessing and tokenization
- **spaCy**: Advanced NLP operations
- **FastAPI**: Backend API server
- **React + Vite**: Frontend web interface
- **scikit-learn**: Similarity computations and metrics
- **NetworkX**: Graph-based algorithms
- **rouge-score**: ROUGE metric computation

### Backend Setup

**Requirements.txt includes**:
```
torch
transformers
fastapi
uvicorn
nltk
spacy
scikit-learn
networkx
rouge-score
python-multipart
pydantic
```

### Frontend Setup

**Tech Stack**:
- React 18+ with Hooks
- Vite for bundling
- Axios for API calls
- Responsive CSS (mobile & desktop)
- Standalone HTML mode (no build required)

---

## Datasets

### 1. CNN/DailyMail Dataset

**Statistics**:
- Training: 287,113 documents
- Validation: 13,368 documents
- Test: 11,490 documents
- Average document length: ~760 words
- Average summary length: ~56 words

**Access**: Available through Hugging Face Datasets

### 2. XSum (Extreme Summarization)

**Statistics**:
- Training: 204,045 documents
- Validation: 11,332 documents
- Test: 11,334 documents
- Average document length: ~431 words
- Average summary length: ~23 words (1 sentence)

**Access**: Available through Hugging Face Datasets

### 3. Optional: Scientific Papers

- **ArXiv Dataset**: Computer science papers with abstracts
- **PubMed**: Biomedical literature with abstracts

---

## Evaluation Metrics

### 1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Primary metric for summarization evaluation:

- **ROUGE-1**: Unigram overlap (content coverage)
- **ROUGE-2**: Bigram overlap (fluency)
- **ROUGE-L**: Longest Common Subsequence (coherence)

**Target Scores**:
- CNN/DailyMail: ROUGE-1 > 42%, ROUGE-2 > 20%, ROUGE-L > 38%
- XSum: ROUGE-1 > 45%, ROUGE-2 > 22%, ROUGE-L > 42%

### 2. BLEU (Bilingual Evaluation Understudy)

Measures n-gram precision with brevity penalty
- **Target Score**: > 0.35 for abstractive quality

### 3. BERTScore

Semantic similarity using contextual embeddings
- **Target F1 Score**: > 0.88

### 4. METEOR

Considers synonyms, stemming, and paraphrasing
- **Target Score**: > 0.25

### 5. Human Evaluation

Qualitative assessment on scales of 1-5:
- Relevance
- Coherence
- Fluency
- Non-redundancy
- Factual Consistency

---

## Installation and Setup

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# pip package manager
pip --version
```

### Backend Installation

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Frontend Installation

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Verify installation
npm --version
```

---

## Running the Application

### Starting the Backend

```bash
cd backend

# On Windows (using provided batch script)
run_backend.bat

# On Linux/Mac
python main.py

# FastAPI will start on http://localhost:8000
# API documentation available at http://localhost:8000/docs
```

### Starting the Frontend

```bash
cd frontend

# Development mode
npm run dev
# Frontend will run on http://localhost:5173

# Production build
npm run build
# Output in dist/
```

### Complete Stack

```bash
# Terminal 1: Backend
cd backend
python main.py

# Terminal 2: Frontend
cd frontend
npm run dev

# Open browser to http://localhost:5173
```

### Standalone Mode

Open `frontend/standalone.html` directly in a browser for a self-contained demo.

---

## Evaluation Results

### Expected Performance

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | METEOR |
|-------|---------|---------|---------|------|--------|
| Pure Extractive (BERT) | 40.2 | 17.6 | 36.7 | 0.10 | 0.22 |
| Pure Abstractive (BART) | 44.2 | 21.3 | 40.8 | 0.34 | 0.27 |
| **Proposed Hybrid** | **46.5** | **23.1** | **43.2** | **0.38** | **0.29** |

### Qualitative Improvements

1. **Better Factual Accuracy** - Extractive phase preserves facts
2. **Enhanced Fluency** - Abstractive phase ensures readability
3. **Reduced Redundancy** - Coverage mechanism prevents repetition
4. **Improved Compression** - Higher information density
5. **Balanced Abstraction** - Avoids over-abstraction

---

## Challenges and Limitations

### Technical Challenges

1. **Computational Complexity**
   - Long documents require significant GPU memory
   - Two-stage pipeline increases inference time
   - *Solution*: Implement efficient chunking and caching

2. **Information Loss**
   - Extractive phase may miss nuanced information
   - *Solution*: Increase extracted sentences for longer documents

3. **Coherence in Hybrid Output**
   - Maintaining logical flow between extracted and generated content
   - *Solution*: Implement coreference resolution in post-processing

4. **Domain Adaptation**
   - Models trained on news may not generalize well
   - *Solution*: Fine-tune on domain-specific datasets

### Limitations

1. **Language Dependency**: Focuses on English text
2. **Document Length**: Very long documents (>5000 words) require hierarchical processing
3. **Real-time Constraints**: Two-stage approach increases latency
4. **Training Data**: Fine-tuning requires large annotated datasets

---

## Future Work

### Short-term Enhancements

1. **Optimization**
   - Model quantization (8-bit) for faster inference
   - Knowledge distillation for smaller models
   - Dynamic sentence selection based on document type

2. **Advanced Features**
   - Query-focused summarization
   - Multi-document summarization
   - Controllable summarization (length, style, detail)

3. **Improved Evaluation**
   - Factual consistency checkers using NLI models
   - Automated coherence scoring
   - Domain-specific benchmarks

### Long-term Research Directions

1. **Multilingual Support** - mBART, mT5, cross-lingual summarization
2. **Hierarchical Processing** - Multi-level abstraction for long documents
3. **Interactive Summarization** - User feedback integration
4. **Multimodal Summarization** - Visual, video, audio content
5. **Reinforcement Learning** - RL-based optimization
6. **Explainability** - Visualization of decision processes

---

## Features

### File Upload Capabilities
- **Supported Formats**: PDF, TXT, DOCX, PPTX
- **OCR Support**: Extract text from images
- See `FILE_UPLOAD_FEATURES.md` for details
- See `backend/OCR_SETUP.md` for OCR configuration

### Web Demo Features
- **Real-time Summarization**: Get summaries within seconds
- **Extractive Display**: See which sentences were selected
- **Multiple Summaries**: Compare different summary lengths
- **Score Display**: View evaluation metrics (if enabled)
- **Responsive Design**: Works on desktop and mobile

---

## Documentation

Refer to the `docs/` folder for detailed information:

- **architecture.md** - Detailed system architecture
- **deployment.md** - Production deployment guide
- **evaluation.md** - Evaluation methodology details
- **QUICKSTART.md** - Get started in 5 minutes
- **FRONTEND_SETUP.md** - Frontend configuration
- **FILE_UPLOAD_FEATURES.md** - File handling capabilities

---

## Quick Links

- **API Documentation**: http://localhost:8000/docs (when backend is running)
- **Frontend Application**: http://localhost:5173 (when frontend is running)
- **Assignment Documentation**: See `docs/assignment1.txt` and `docs/assignment2.txt`

---

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```
Solution: Reduce batch size, use gradient accumulation, or implement model quantization
```

**Slow Inference**
```
Solution: Use smaller models (t5-small, bart-base), reduce max_length, or use GPU
```

**Poor Summary Quality**
```
Solution: Adjust number of extracted sentences, tune beam search parameters, or fine-tune on domain-specific data
```

**Frontend Connection Issues**
```
Solution: Set VITE_API_BASE_URL environment variable if backend runs on different address
Example: VITE_API_BASE_URL=http://your-backend-ip:8000 npm run dev
```

---

## References

### Key Papers

1. See, A., Liu, P. J., & Manning, C. D. (2017). "Get To The Point: Summarization with Pointer-Generator Networks." ACL 2017.

2. Liu, Y., & Lapata, M. (2019). "Text Summarization with Pretrained Encoders." EMNLP-IJCNLP 2019.

3. Lewis, M., et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training." ACL 2020.

4. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR 2020.

5. Zhang, J., et al. (2020). "PEGASUS: Pre-training with Extracted Gap-sentences." ICML 2020.

6. Divya, S., et al. (2024). "Unified extractive-abstractive summarization." PeerJ Computer Science.

7. Ahmed, R., & Hemanth, D. J. (2025). "Hybrid text summarization: Integrating extractive and abstractive models." Intelligent Decision Technologies.

8. Yadav, D., Desai, J., & Yadav, A. K. (2022). "Automatic Text Summarization Methods: A Comprehensive Review." arXiv:2204.01849.

### Datasets

9. Hermann, K. M., et al. (2015). "Teaching machines to read and comprehend." NIPS 2015. (CNN/DailyMail)

10. Narayan, S., Cohen, S. B., & Lapata, M. (2018). "Don't Give Me the Details, Just the Summary!" EMNLP 2018. (XSum)

### Evaluation

11. Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." ACL 2004.

12. Zhang, T., et al. (2020). "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.

---

## Contact and Contribution

**Author**: Chaudhary6559  
**Repository**: Automatic-Text-Summarization-with-Hybrid-Extractive-Abstractive-Methods  
**Course**: Natural Language Processing

For questions, suggestions, or collaboration opportunities, please open an issue or pull request.

---

**Last Updated**: October 2026  
**Version**: 2.0

---

## License

This project is part of an educational course. Please refer to the repository for the specific license.

---

## Acknowledgments

This project builds upon the work of numerous researchers in the field of natural language processing and text summarization. We acknowledge the Hugging Face community for providing pre-trained models and datasets.
