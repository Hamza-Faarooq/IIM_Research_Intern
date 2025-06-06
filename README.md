# IIM_Research_Intern
# 🎥 EviCLIP: Multimodal Review Helpfulness Prediction
**CLIP-Based Semantic Alignment with Evidential Fusion**  
*Predicting IMDB Review Helpfulness Using Text and Movie Stills*

[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="assets/pipeline.png" alt="Architecture Diagram" width="800"/>
  <p>Project Architecture: CLIP-based alignment + Multimodal fusion</p>
</div>

## 📖 Table of Contents
- [Key Features](#-key-features)
- [Performance Highlights](#-performance-highlights)
- [Installation](#-installation)
- [Usage](#-usage)
- [Directory Structure](#-directory-structure)
- [Dataset](#-dataset)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

## 🚀 Key Features
- **Dynamic Frame Selection**: CLIP-powered semantic alignment between reviews and movie stills
- **Uncertainty-Aware Fusion**: Dempster-Shafer evidential fusion of text/image features
- **Multimodal Benchmarking**: Compare 3 fusion strategies (Evidential/Attention/Simple)
- **State-of-the-Art Metrics**: Achieves **0.9823 NDCG** and **0.8990 MAP** on balanced dataset
- **Production-Ready Pipeline**: End-to-end workflow from data processing to deployment

## 📈 Performance Highlights
| Metric       | Evidential Fusion | Attention Fusion | Simple Fusion |
|--------------|-------------------|------------------|---------------|
| **Accuracy** | 0.8053            | 0.7917           | 0.7902        |
| **F1**       | 0.8013            | 0.7964           | 0.7851        |
| **AUC-ROC**  | 0.8836            | 0.8721           | 0.8618        |
| **MAP**      | 0.8990            | 0.8478           | 0.8323        |
| **NDCG**     | 0.9823            | 0.9214           | 0.9107        |

## 💻 Installation
### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 11.7
- 16GB+ RAM



## 📊 Dataset
### IMDB Reviews + Movie Stills
- **4,017** curated IMDB reviews with helpfulness labels
- **1,443** high-resolution frames from 15 sci-fi movies
- Balanced 50-50 split between helpful/unhelpful reviews

Dataset Structure:

