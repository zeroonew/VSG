<div align="center">

# 🎯 VSG: Voxel-based Subspace Grid

### A Universal Tool for Dataset Coverage and Similarity Measurement in Embedding Spaces

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

**Quantify Dataset Diversity · Evaluate Generative Models · Analyze Embedding Space Coverage**

**Works with any modality: Images | Text | Audio | Molecules | Any Embedding**

</div>

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [API Documentation](#-api-documentation)
- [Applications](#-applications)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Introduction

**VSG (Voxel-based Subspace Grid)** is an efficient tool for quantifying dataset coverage and similarity in embedding spaces. By partitioning high-dimensional feature spaces into voxel grids, VSG enables:

- 🎯 **Precise Measurement** of dataset coverage in embedding spaces
- 🔄 **Comparison** of semantic overlap between different datasets
- 📊 **Evaluation** of data diversity in generative models
- ⚡ **Efficient Computation** supporting large-scale datasets
- 🌐 **Universal Application** to any data that can be represented as embeddings

### Core Idea

VSG partitions high-dimensional embedding spaces into multiple low-dimensional subspaces, discretizing each subspace using fixed-size voxels. Coverage is measured by counting occupied voxels, while similarity is quantified through the Intersection over Union (IoU) of voxel sets.

### Multi-Modality Support

VSG is **modality-agnostic** and works with any fixed-size embedding vectors. This package provides built-in support for image embeddings (CLIP, DINOv2), but the algorithm can be applied to embeddings from any modality (text, audio, etc.) by simply passing your embedding matrix to the VSG class.

---

## 🚀 Key Features

### 🔬 Precise Coverage Measurement
- Voxel-based spatial discretization
- Configurable subspace dimensions
- Multi-scale analysis for robust estimation

### 🎨 Flexible Similarity Analysis
- **IoU (Intersection over Union)**: Symmetric similarity measure
- **AinB**: Coverage of A in B (asymmetric)
- **BinA**: Coverage of B in A (asymmetric)
- **Multi-scale Integration**: Computed and integrated across multiple voxel sizes

### 🌐 Universal Embedding Support
- Built-in support for **CLIP** and **DINO/DINOv2** image models
- Modality-agnostic design — works with any embedding vectors
- Automatic caching to avoid redundant computation
- Support for single-model or joint embeddings

### ⚡ Efficient Implementation
- NumPy vectorized operations
- 1D encoding for accelerated voxel deduplication
- Supports incremental coverage computation

---

## 📦 Installation

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/vsg.git
cd vsg

# Install dependencies
pip install -r requirements.txt

# Install VSG
pip install -e .
```

### Dependencies

- Python >= 3.7
- NumPy >= 1.20.0
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- transformers >= 4.20.0
- clip >= 1.0
- Pillow >= 8.0.0
- tqdm >= 4.60.0

---

## 🎯 Quick Start

### Example 1: Compute Dataset Coverage

```python
import numpy as np
from vsg import VSG

# Generate data (or load your embeddings)
X = np.random.randn(1000, 128)

# Initialize VSG
vsg = VSG(subspace_dim=1, voxel_size=0.1)

# Compute coverage
coverage = vsg.compute_coverage(X)
print(f"Dataset coverage: {coverage:.2f}")
```

### Example 2: Compare Similarity Between Two Datasets

```python
import numpy as np
from vsg import VSG

# Generate two datasets (or load your embeddings)
X_A = np.random.randn(1000, 128) * 0.5
X_B = np.random.randn(1000, 128) * 0.5 + 0.3

# Initialize VSG
vsg = VSG(subspace_dim=1, voxel_size=0.1)

# Compute similarity
similarity = vsg.compute_similarity_iou(X_A, X_B)

print(f"IoU:  {similarity['iou']:.4f}")
print(f"AinB: {similarity['a_in_b']:.4f}")
print(f"BinA: {similarity['b_in_a']:.4f}")
```

### Example 3: Extract Image Embeddings

```python
from vsg import EmbeddingExtractor

# Initialize embedding extractor
extractor = EmbeddingExtractor(cache_dir='./cache')

# Extract CLIP embeddings from image folder
result = extractor.extract_image_embeddings(
    image_dir='./images/dataset_A',
    model_type='clip',
    model_name='ViT-B/32'
)

print(f"Embedding shape: {result['embeddings'].shape}")
print(f"Extracted {len(result['filenames'])} images")
```

### Example 4: Complete Pipeline - Embedding Extraction + VSG Analysis

```python
from vsg import EmbeddingExtractor, VSG

# 1. Extract embeddings
extractor = EmbeddingExtractor(cache_dir='./cache')

result_A = extractor.extract_image_embeddings(
    './images/dataset_A',
    model_type='clip',
    model_name='ViT-B/32'
)

result_B = extractor.extract_image_embeddings(
    './images/dataset_B',
    model_type='clip',
    model_name='ViT-B/32'
)

# 2. Compute VSG similarity
vsg = VSG(subspace_dim=1, voxel_size=0.1)
similarity = vsg.compute_similarity_iou(
    result_A['embeddings'],
    result_B['embeddings']
)

print(f"Dataset similarity (IoU): {similarity['iou']:.4f}")
```

---

## 📚 Usage Examples

### Multi-Scale Similarity Analysis

```python
from vsg import VSG
import numpy as np

X_A = np.random.randn(1000, 128)
X_B = np.random.randn(1000, 128)

vsg = VSG(subspace_dim=1, voxel_size=0.1)

# Multi-scale analysis
similarity = vsg.compute_similarity_iou_multiscale(
    X_A, X_B,
    voxel_sizes=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
)

print(f"Multi-scale IoU: {similarity['iou']:.4f}")
print(f"IoU curve across scales: {similarity['iou_curve']}")
```

### Subspace Coverage Analysis

```python
from vsg import VSG
import numpy as np

X = np.random.randn(1000, 128)
vsg = VSG(subspace_dim=1, voxel_size=0.1)

# Compute coverage for each subspace
coverages = vsg.compute_coverage_per_subspace(X)

print(f"Average coverage: {coverages.mean():.2f}")
print(f"Coverage std: {coverages.std():.2f}")
print(f"Subspace with min coverage: {coverages.argmin()}")
print(f"Subspace with max coverage: {coverages.argmax()}")
```

### Extract Joint Embeddings (CLIP + DINO)

```python
from vsg import EmbeddingExtractor

extractor = EmbeddingExtractor(cache_dir='./cache')

# Extract and concatenate CLIP and DINO embeddings
joint_result = extractor.extract_joint_image_embeddings(
    image_dir='./images/dataset_A',
    clip_model='ViT-B/32',
    dino_model='facebook/dinov2-vit-base-p14'
)

print(f"CLIP embeddings: {joint_result['clip_embeddings'].shape}")
print(f"DINO embeddings: {joint_result['dino_embeddings'].shape}")
print(f"Joint embeddings: {joint_result['joint_embeddings'].shape}")
```

---

## 🔧 API Documentation

### VSG Class

#### Initialization Parameters

```python
VSG(subspace_dim=1, voxel_size=0.1)
```

- `subspace_dim` (int): Dimension of each subspace, default 1
- `voxel_size` (float): Voxel size, default 0.1

#### Main Methods

##### `compute_coverage(X)`

Compute coverage of a single dataset

**Parameters:**
- `X` (np.ndarray): Embedding matrix, shape (n_samples, n_features)

**Returns:**
- `coverage` (float): Coverage value

##### `compute_similarity_iou(X_A, X_B)`

Compute IoU similarity between two datasets

**Parameters:**
- `X_A` (np.ndarray): Embedding matrix of dataset A
- `X_B` (np.ndarray): Embedding matrix of dataset B

**Returns:**
- `dict`: Dictionary containing:
  - `'iou'`: IoU similarity
  - `'a_in_b'`: Coverage of A in B
  - `'b_in_a'`: Coverage of B in A

##### `compute_similarity_iou_multiscale(X_A, X_B, voxel_sizes=None)`

Multi-scale IoU similarity analysis

**Parameters:**
- `X_A` (np.ndarray): Dataset A
- `X_B` (np.ndarray): Dataset B
- `voxel_sizes` (list): List of voxel sizes, default [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

**Returns:**
- `dict`: Multi-scale analysis results and curve data

### EmbeddingExtractor Class

#### Initialization Parameters

```python
EmbeddingExtractor(cache_dir='./cache')
```

- `cache_dir` (str): Cache directory path

#### Main Methods

##### `extract_image_embeddings(image_dir, model_type, model_name, ...)`

Extract embeddings from image folder

**Parameters:**
- `image_dir` (str): Path to image folder
- `model_type` (str): Model type ('clip' or 'dino')
- `model_name` (str): Model name
- `batch_size` (int): Batch size, default 32
- `num_workers` (int): Number of data loading workers, default 0
- `force_extract` (bool): Force re-extraction, default False
- `image_size` (int): Image size, default 224

**Returns:**
- `dict`: Contains 'embeddings', 'filenames', 'model_type', 'model_name'

##### `extract_text_embeddings(texts, model_name, ...)`

Extract embeddings from text data

**Parameters:**
- `texts` (list): List of text strings
- `model_name` (str): Sentence-transformers model name
- `batch_size` (int): Batch size, default 32
- `force_extract` (bool): Force re-extraction, default False
- `cache_key` (str): Cache identifier

**Returns:**
- `dict`: Contains 'embeddings', 'texts', 'model_name'

##### `extract_custom_embeddings(data, encoder_fn, ...)`

Extract embeddings using custom encoder

**Parameters:**
- `data`: Input data (any format)
- `encoder_fn`: Function that takes data and returns embeddings
- `cache_key` (str): Cache identifier
- `model_name` (str): Model identifier
- `force_extract` (bool): Force re-extraction, default False

**Returns:**
- `dict`: Contains 'embeddings', 'model_name'

##### `extract_joint_image_embeddings(image_dir, clip_model, dino_model, ...)`

Extract and concatenate CLIP and DINO embeddings

**Parameters:**
- `image_dir` (str): Path to image folder
- `clip_model` (str): CLIP model name
- `dino_model` (str): DINO model name
- Other parameters same as `extract_image_embeddings`

**Returns:**
- `dict`: Contains 'clip_embeddings', 'dino_embeddings', 'joint_embeddings', 'filenames'

---

## 🎨 Applications

### 1. Generative Model Evaluation

Evaluate data diversity of GANs, VAEs, diffusion models, etc.:

```python
# Compare coverage of real and generated datasets
coverage_real = vsg.compute_coverage(embeddings_real)
coverage_generated = vsg.compute_coverage(embeddings_generated)

# Compute similarity
similarity = vsg.compute_similarity_iou(embeddings_real, embeddings_generated)
```

### 2. Dataset Diversity Analysis

Analyze diversity differences across datasets:

```python
# Compare coverage of multiple datasets
for dataset_name, embeddings in datasets.items():
    coverage = vsg.compute_coverage(embeddings)
    print(f"{dataset_name}: {coverage:.2f}")
```

### 3. Data Augmentation Effectiveness

Evaluate the effectiveness of data augmentation strategies:

```python
# Compare coverage of original and augmented data
coverage_original = vsg.compute_coverage(embeddings_original)
coverage_augmented = vsg.compute_coverage(embeddings_augmented)

# Compute overlap
overlap = vsg.compute_similarity_iou(embeddings_original, embeddings_augmented)
```

### 4. Embedding Space Exploration

Analyze coverage across different feature subspaces:

```python
# Analyze coverage of each subspace
coverages = vsg.compute_coverage_per_subspace(embeddings)

# Find subspaces with lowest and highest coverage
print(f"Subspace with lowest coverage: {coverages.argmin()}")
print(f"Subspace with highest coverage: {coverages.argmax()}")
```

### 5. Cross-Modality Comparison

Compare datasets across different modalities using shared embedding spaces:

```python
# Extract embeddings from different modalities
image_embeddings = extractor.extract_image_embeddings('./images', model_type='clip')
text_embeddings = extractor.extract_text_embeddings(texts, model_name='clip-text')

# Compare in shared CLIP embedding space
similarity = vsg.compute_similarity_iou(
    image_embeddings['embeddings'],
    text_embeddings['embeddings']
)
```

---

## 📊 Parameter Tuning Guide

### subspace_dim (Subspace Dimension)

- **Recommended**: 1-8
- **Smaller values (1-2)**: Finer-grained analysis, higher computational cost
- **Larger values (4-8)**: Coarser-grained analysis, lower computational cost
- **Suggestion**: Start with 1, adjust based on computational resources and requirements

### voxel_size (Voxel Size)

- **Recommended**: 0.05-0.5
- **Smaller values (0.05-0.1)**: Finer coverage estimation, sensitive to noise
- **Larger values (0.2-0.5)**: Smoother estimation, may lose details
- **Suggestion**: Use multi-scale analysis (`compute_similarity_iou_multiscale`) for robust results

---

## 🔗 Citation

If you use VSG in your research, please cite:

```bibtex
@software{vsg2024,
  title={VSG: Voxel-based Subspace Grid for Dataset Coverage and Similarity Analysis in Embedding Spaces},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/vsg}
}
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, bug reports, and improvement suggestions are welcome!

---

## 📧 Contact

For questions or suggestions, please contact:
- Email: your.email@example.com
- GitHub Issues: https://github.com/yourusername/vsg/issues

---

<div align="center">

**🌟 If this project helps you, please give us a Star!**

</div>
