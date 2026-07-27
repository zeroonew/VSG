"""
VSG (Voxel-based Subspace Grid) - Coverage and Similarity Metrics for Embedding Spaces

Core Features:
1. compute_coverage(X) - Compute coverage of a single dataset
2. compute_similarity_iou(X_A, X_B) - Compute IoU similarity between two datasets
3. compute_similarity_iou_multiscale(X_A, X_B) - Multi-scale IoU similarity

Key Characteristics:
- No PCA transformation, operates directly on original embeddings
- Configurable fixed voxel size
- Supports both single-set coverage and pairwise similarity computation
- Works with any modality that can be represented as embeddings (images, text, audio, etc.)
"""

import numpy as np
from collections import defaultdict


class VSG:
    """
    VSG: Voxel-based Subspace Grid
    
    A tool for measuring dataset coverage and similarity in embedding spaces.
    Works with any modality that can be represented as dense embeddings.
    """

    def __init__(self, subspace_dim=1, voxel_size=0.1):
        """
        Initialize VSG
        
        Args:
            subspace_dim: Dimension of each subspace (default: 1)
            voxel_size: Fixed voxel size for discretization (default: 0.1)
        """
        self.subspace_dim = subspace_dim
        self.voxel_size = voxel_size
        self.n_subspaces_ = None

    @staticmethod
    def _encode_voxels_1d(voxels):
        """
        Encode multi-dimensional voxel indices to single integers for fast 1D deduplication
        
        Args:
            voxels: Voxel indices (n_samples, dim)
        
        Returns:
            codes: Encoded single integer array
        """
        # Shift each column to make it non-negative
        mins = voxels.min(axis=0)
        shifted = voxels - mins
        ranges = shifted.max(axis=0) + 1

        # Compute weights for each column (mixed-radix encoding)
        dim = voxels.shape[1]
        weights = np.ones(dim, dtype=np.int64)
        for d in range(1, dim):
            weights[d] = weights[d - 1] * ranges[d - 1]

        return shifted @ weights

    def compute_coverage(self, X):
        """
        Compute coverage of a dataset in embedding space
        
        Args:
            X: Embedding matrix (n_samples, n_features)
        
        Returns:
            coverage: Average coverage across all subspaces
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Compute number of subspaces
        self.n_subspaces_ = n_features // self.subspace_dim
        if self.n_subspaces_ == 0:
            raise ValueError(f"Feature dimension {n_features} is smaller than subspace dimension {self.subspace_dim}")

        total_coverage = 0

        for i in range(self.n_subspaces_):
            # Extract subspace
            start = i * self.subspace_dim
            end = start + self.subspace_dim
            sub = X[:, start:end]

            # Compute voxel indices (fixed voxel size)
            voxel_indices = np.floor(sub / self.voxel_size).astype(np.int64)

            # Encode to 1D integers for fast deduplication
            codes = self._encode_voxels_1d(voxel_indices)
            n_occupied = len(np.unique(codes))
            total_coverage += n_occupied

        # Equal-weighted average
        coverage = total_coverage / self.n_subspaces_

        return coverage

    def compute_coverage_per_subspace(self, X):
        """
        Compute coverage for each subspace individually
        
        Args:
            X: Embedding matrix (n_samples, n_features)
        
        Returns:
            coverages: Coverage array for each subspace (n_subspaces,)
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Compute number of subspaces
        self.n_subspaces_ = n_features // self.subspace_dim
        if self.n_subspaces_ == 0:
            raise ValueError(f"Feature dimension {n_features} is smaller than subspace dimension {self.subspace_dim}")

        coverages = []

        for i in range(self.n_subspaces_):
            # Extract subspace
            start = i * self.subspace_dim
            end = start + self.subspace_dim
            sub = X[:, start:end]

            # Compute voxel indices (fixed voxel size)
            voxel_indices = np.floor(sub / self.voxel_size).astype(np.int64)

            # Encode to 1D integers for fast deduplication
            codes = self._encode_voxels_1d(voxel_indices)
            n_occupied = len(np.unique(codes))
            coverages.append(n_occupied)

        return np.array(coverages)

    @staticmethod
    def _encode_voxels(voxels_A, voxels_B):
        """
        Encode voxel indices from two datasets to single integers
        Uses unified encoding parameters (global min and ranges) to ensure 
        identical voxels receive identical codes
        
        Args:
            voxels_A: Voxel indices from dataset A (n_A, dim)
            voxels_B: Voxel indices from dataset B (n_B, dim)
        
        Returns:
            codes_A, codes_B: Encoded single integer arrays
        """
        # Use global min and ranges from both A and B combined
        combined_min = np.minimum(voxels_A.min(axis=0), voxels_B.min(axis=0))
        combined_max = np.maximum(voxels_A.max(axis=0), voxels_B.max(axis=0))

        shifted_A = voxels_A - combined_min
        shifted_B = voxels_B - combined_min

        ranges = (combined_max - combined_min) + 1

        # Compute weights for each column (mixed-radix encoding)
        dim = voxels_A.shape[1]
        weights = np.ones(dim, dtype=np.int64)
        for d in range(1, dim):
            weights[d] = weights[d - 1] * ranges[d - 1]

        codes_A = shifted_A @ weights
        codes_B = shifted_B @ weights

        return codes_A, codes_B

    def compute_similarity_iou(self, X_A, X_B):
        """
        Compute IoU, AinB, and BinA similarity between two datasets
        Uses integer encoding + 1D np.unique to avoid Python object creation
        
        Args:
            X_A: Embeddings from dataset A
            X_B: Embeddings from dataset B
        
        Returns:
            dict: Dictionary containing three metrics:
                - 'iou': IoU similarity (intersection / union)
                - 'a_in_b': Coverage of A in B (intersection / |A|)
                - 'b_in_a': Coverage of B in A (intersection / |B|)
        """
        X_A = np.asarray(X_A, dtype=np.float64)
        X_B = np.asarray(X_B, dtype=np.float64)

        n_features = X_A.shape[1]
        assert X_B.shape[1] == n_features, "Both datasets must have the same feature dimension"

        n_subspaces = n_features // self.subspace_dim
        if n_subspaces == 0:
            raise ValueError(f"Feature dimension {n_features} is smaller than subspace dimension {self.subspace_dim}")

        total_iou = 0
        total_ainb = 0
        total_bina = 0

        for i in range(n_subspaces):
            start = i * self.subspace_dim
            end = start + self.subspace_dim

            sub_A = X_A[:, start:end]
            sub_B = X_B[:, start:end]

            # Compute voxel indices
            voxels_A = np.floor(sub_A / self.voxel_size).astype(np.int64)
            voxels_B = np.floor(sub_B / self.voxel_size).astype(np.int64)

            # Encode to single integers using unified parameters
            codes_A, codes_B = self._encode_voxels(voxels_A, voxels_B)

            # Get unique voxel codes
            unique_A = np.unique(codes_A)
            unique_B = np.unique(codes_B)

            n_A = len(unique_A)
            n_B = len(unique_B)

            # Merge and count occurrences
            combined = np.concatenate([unique_A, unique_B])
            _, counts = np.unique(combined, return_counts=True)

            # Codes appearing >= 2 times are in the intersection
            intersection = int(np.sum(counts >= 2))
            union = n_A + n_B - intersection

            # IoU = intersection / union
            iou = intersection / union if union > 0 else 0

            # AinB = intersection / |A|
            ainb = intersection / n_A if n_A > 0 else 0

            # BinA = intersection / |B|
            bina = intersection / n_B if n_B > 0 else 0

            total_iou += iou
            total_ainb += ainb
            total_bina += bina

        return {
            'iou': total_iou / n_subspaces,
            'a_in_b': total_ainb / n_subspaces,
            'b_in_a': total_bina / n_subspaces
        }

    def compute_similarity_iou_multiscale(self, X_A, X_B, voxel_sizes=None):
        """
        Compute IoU, AinB, and BinA at multiple voxel sizes, then use trapezoidal 
        integration (similar to AUROC) to compute weighted average across scales.
        
        Args:
            X_A: Embeddings from dataset A
            X_B: Embeddings from dataset B
            voxel_sizes: List of voxel sizes, e.g., [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
                         If None, uses default 6 scales
        
        Returns:
            dict: Dictionary containing multi-scale weighted metrics:
                - 'iou': Multi-scale IoU area (trapezoidal integration normalized)
                - 'a_in_b': Multi-scale AinB area
                - 'b_in_a': Multi-scale BinA area
                - 'iou_curve': IoU values at each scale
                - 'a_in_b_curve': AinB values at each scale
                - 'b_in_a_curve': BinA values at each scale
                - 'voxel_sizes': List of scales used
        """
        if voxel_sizes is None:
            # Default: 6 logarithmically spaced scales
            voxel_sizes = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        # Ensure scales are sorted
        voxel_sizes = sorted(voxel_sizes)
        
        iou_curve = []
        ainb_curve = []
        bina_curve = []
        
        # Compute single-scale IoU at each voxel size
        for vs in voxel_sizes:
            # Temporarily modify voxel_size
            original_voxel_size = self.voxel_size
            self.voxel_size = vs
            
            result = self.compute_similarity_iou(X_A, X_B)
            iou_curve.append(result['iou'])
            ainb_curve.append(result['a_in_b'])
            bina_curve.append(result['b_in_a'])
            
            # Restore original voxel_size
            self.voxel_size = original_voxel_size
        
        iou_curve = np.array(iou_curve)
        ainb_curve = np.array(ainb_curve)
        bina_curve = np.array(bina_curve)
        voxel_sizes_arr = np.array(voxel_sizes)
        
        # Use trapezoidal integration to compute area under curve (similar to AUROC)
        # Then divide by scale range to get normalized average IoU
        scale_range = voxel_sizes_arr[-1] - voxel_sizes_arr[0]
        
        if scale_range > 0:
            iou_area = np.trapz(iou_curve, voxel_sizes_arr) / scale_range
            ainb_area = np.trapz(ainb_curve, voxel_sizes_arr) / scale_range
            bina_area = np.trapz(bina_curve, voxel_sizes_arr) / scale_range
        else:
            iou_area = iou_curve[0]
            ainb_area = ainb_curve[0]
            bina_area = bina_curve[0]
        
        return {
            'iou': float(iou_area),
            'a_in_b': float(ainb_area),
            'b_in_a': float(bina_area),
            'iou_curve': iou_curve.tolist(),
            'a_in_b_curve': ainb_curve.tolist(),
            'b_in_a_curve': bina_curve.tolist(),
            'voxel_sizes': voxel_sizes,
        }
