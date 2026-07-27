"""
VSG (Voxel-based Subspace Grid) - Coverage and Similarity Metrics for Embedding Spaces

Core Modules:
- vsg: VSG core algorithm for computing coverage and similarity in embedding spaces
- feature_extractor: Image embedding extraction tool with CLIP and DINOv2 support

VSG is modality-agnostic and works with any fixed-size embedding vectors.
This package provides built-in image embedding extractors, but the VSG algorithm
can be applied to embeddings from any modality (text, audio, etc.).
"""

from .vsg import VSG
from .feature_extractor import EmbeddingExtractor

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = ['VSG', 'EmbeddingExtractor']
