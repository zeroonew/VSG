"""
VSG Usage Examples

Demonstrates how to use the VSG toolkit for dataset coverage and similarity analysis
across multiple modalities (images, text, audio, etc.)
"""

import numpy as np
from vsg import VSG, EmbeddingExtractor


def example_1_synthetic_data():
    """
    Example 1: Synthetic data - VSG basic functionality
    """
    print("=" * 60)
    print("Example 1: Synthetic Data - VSG Basic Functionality")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 128
    
    # Dataset A: Gaussian distribution
    X_A = np.random.randn(n_samples, n_features) * 0.5
    
    # Dataset B: Partially overlapping with A
    X_B = np.random.randn(n_samples, n_features) * 0.5 + 0.3
    
    # Initialize VSG
    vsg = VSG(subspace_dim=1, voxel_size=0.1)
    
    # Compute coverage for individual datasets
    coverage_A = vsg.compute_coverage(X_A)
    coverage_B = vsg.compute_coverage(X_B)
    
    print(f"\nDataset A coverage: {coverage_A:.2f}")
    print(f"Dataset B coverage: {coverage_B:.2f}")
    
    # Compute similarity between two datasets
    similarity = vsg.compute_similarity_iou(X_A, X_B)
    
    print(f"\nSimilarity analysis:")
    print(f"  IoU:   {similarity['iou']:.4f}")
    print(f"  AinB:  {similarity['a_in_b']:.4f}")
    print(f"  BinA:  {similarity['b_in_a']:.4f}")
    
    # Multi-scale similarity
    similarity_multi = vsg.compute_similarity_iou_multiscale(X_A, X_B)
    
    print(f"\nMulti-scale similarity:")
    print(f"  IoU:   {similarity_multi['iou']:.4f}")
    print(f"  AinB:  {similarity_multi['a_in_b']:.4f}")
    print(f"  BinA:  {similarity_multi['b_in_a']:.4f}")
    
    print("\n" + "=" * 60 + "\n")


def example_2_image_embeddings():
    """
    Example 2: Extract image embeddings and compute similarity
    """
    print("=" * 60)
    print("Example 2: Image Embedding Extraction & Similarity Analysis")
    print("=" * 60)
    
    # Initialize embedding extractor
    extractor = EmbeddingExtractor(cache_dir='./cache')
    
    # Assume you have two image folders
    # folder_A = './images/dataset_A'
    # folder_B = './images/dataset_B'
    
    # Extract CLIP embeddings
    # result_A = extractor.extract_image_embeddings(
    #     folder_A,
    #     model_type='clip',
    #     model_name='ViT-B/32',
    #     batch_size=32
    # )
    
    # result_B = extractor.extract_image_embeddings(
    #     folder_B,
    #     model_type='clip',
    #     model_name='ViT-B/32',
    #     batch_size=32
    # )
    
    # Extract joint CLIP + DINO embeddings
    # joint_A = extractor.extract_joint_image_embeddings(
    #     folder_A,
    #     clip_model='ViT-B/32',
    #     dino_model='facebook/dinov2-vit-base-p14'
    # )
    
    # Use extracted embeddings to compute VSG
    # vsg = VSG(subspace_dim=1, voxel_size=0.1)
    
    # # Using CLIP embeddings
    # similarity_clip = vsg.compute_similarity_iou(
    #     result_A['embeddings'],
    #     result_B['embeddings']
    # )
    
    # # Using joint embeddings
    # similarity_joint = vsg.compute_similarity_iou(
    #     joint_A['joint_embeddings'],
    #     joint_B['joint_embeddings']
    # )
    
    print("\nNote: Uncomment the code above and provide actual image folder paths")
    print("The example demonstrates how to:")
    print("  1. Extract CLIP embeddings from image folders")
    print("  2. Extract joint CLIP + DINO embeddings")
    print("  3. Compute VSG similarity using extracted embeddings")
    
    print("\n" + "=" * 60 + "\n")


def example_3_subspace_analysis():
    """
    Example 3: Subspace coverage analysis
    """
    print("=" * 60)
    print("Example 3: Subspace Coverage Analysis")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(1000, 128) * 0.5
    
    # Initialize VSG
    vsg = VSG(subspace_dim=1, voxel_size=0.1)
    
    # Compute coverage for each subspace
    coverages = vsg.compute_coverage_per_subspace(X)
    
    print(f"\nTotal subspaces: {len(coverages)}")
    print(f"Average coverage: {coverages.mean():.2f}")
    print(f"Coverage std: {coverages.std():.2f}")
    print(f"Min coverage: {coverages.min():.2f} (subspace {coverages.argmin()})")
    print(f"Max coverage: {coverages.max():.2f} (subspace {coverages.argmax()})")
    
    # Analyze coverage distribution
    print(f"\nCoverage distribution:")
    print(f"  10th percentile: {np.percentile(coverages, 10):.2f}")
    print(f"  Median: {np.median(coverages):.2f}")
    print(f"  90th percentile: {np.percentile(coverages, 90):.2f}")
    
    print("\n" + "=" * 60 + "\n")


def example_4_parameter_tuning():
    """
    Example 4: Parameter tuning - Impact of subspace dimension and voxel size
    """
    print("=" * 60)
    print("Example 4: Parameter Tuning")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X_A = np.random.randn(1000, 128) * 0.5
    X_B = np.random.randn(1000, 128) * 0.5 + 0.3
    
    print("\nIoU similarity under different parameter configurations:")
    print("-" * 60)
    
    # Test different subspace dimensions
    for subspace_dim in [1, 2, 4, 8]:
        vsg = VSG(subspace_dim=subspace_dim, voxel_size=0.1)
        similarity = vsg.compute_similarity_iou(X_A, X_B)
        print(f"subspace_dim={subspace_dim:2d}, voxel_size=0.1  ->  IoU={similarity['iou']:.4f}")
    
    print()
    
    # Test different voxel sizes
    for voxel_size in [0.05, 0.1, 0.2, 0.5]:
        vsg = VSG(subspace_dim=1, voxel_size=voxel_size)
        similarity = vsg.compute_similarity_iou(X_A, X_B)
        print(f"subspace_dim= 1, voxel_size={voxel_size:.2f}  ->  IoU={similarity['iou']:.4f}")
    
    print("\n" + "=" * 60 + "\n")


def example_5_batch_comparison():
    """
    Example 5: Batch comparison of multiple datasets
    """
    print("=" * 60)
    print("Example 5: Batch Dataset Comparison")
    print("=" * 60)
    
    # Generate multiple datasets
    np.random.seed(42)
    datasets = {}
    
    for i in range(5):
        X = np.random.randn(500, 64) * (0.3 + 0.1 * i)
        datasets[f'Dataset_{i+1}'] = X
    
    # Initialize VSG
    vsg = VSG(subspace_dim=1, voxel_size=0.1)
    
    # Compute coverage for all datasets
    print("\nCoverage for each dataset:")
    print("-" * 60)
    for name, X in datasets.items():
        coverage = vsg.compute_coverage(X)
        print(f"{name:15s}: {coverage:.2f}")
    
    # Compute similarity for all dataset pairs
    print("\nDataset pair similarity (IoU):")
    print("-" * 60)
    
    dataset_names = list(datasets.keys())
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            name_A = dataset_names[i]
            name_B = dataset_names[j]
            
            similarity = vsg.compute_similarity_iou(
                datasets[name_A],
                datasets[name_B]
            )
            
            print(f"{name_A} vs {name_B}: IoU={similarity['iou']:.4f}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == '__main__':
    # Run all examples
    example_1_synthetic_data()
    example_2_image_embeddings()
    example_3_subspace_analysis()
    example_4_parameter_tuning()
    example_5_batch_comparison()
    
    print("\n✓ All examples completed!")
    print("\nTips:")
    print("  - Adjust parameters (subspace_dim, voxel_size) to fit your data")
    print("  - Use EmbeddingExtractor to extract image embeddings")
    print("  - Multi-scale analysis provides more robust similarity estimates")
    print("  - VSG works with any modality that can be represented as embeddings")
