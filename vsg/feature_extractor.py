"""
Image Embedding Extraction Tool

Provides pre-built extractors for image embeddings using CLIP and DINOv2 models.
VSG itself is modality-agnostic — any fixed-size embedding vector can be used.

Features:
- Automatic caching to avoid redundant computation
- Support for batch processing
- CLIP and DINOv2 image encoders included
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path


class ImageFolderDataset(Dataset):
    """
    Generic image folder dataset
    Loads images from any folder structure
    """
    
    def __init__(self, image_dir, transform=None, extensions=None):
        """
        Args:
            image_dir: Path to image folder
            transform: Image transformation pipeline
            extensions: List of supported image extensions
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Collect all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.image_dir.glob(f'*{ext}'))
            self.image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        self.image_paths = sorted(list(set(self.image_paths)))  # Deduplicate and sort
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformation
        if self.transform is not None:
            image = self.transform(image)
        
        # Return image and filename (as label)
        return image, str(img_path.name)


class CLIPImageEncoder:
    """
    CLIP Image Encoder
    Extracts visual embeddings using CLIP model
    """
    
    def __init__(self, model_name='ViT-B/32', device=None):
        """
        Args:
            model_name: CLIP model name (e.g., 'ViT-B/32', 'ViT-L/14')
            device: Computing device ('cuda' or 'cpu')
        """
        import clip
        
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
    
    def encode(self, dataloader):
        """
        Extract embeddings from dataloader
        
        Args:
            dataloader: PyTorch DataLoader yielding (images, filenames)
        
        Returns:
            embeddings: Embedding matrix (n_samples, embedding_dim)
            filenames: List of filenames
        """
        all_embeddings = []
        all_filenames = []
        
        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc=f"Extracting CLIP image embeddings"):
                images = images.to(self.device)
                embeddings = self.model.encode_image(images)
                
                # L2 normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_filenames.extend(filenames)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings, all_filenames


class DINOV2Encoder:
    """
    DINOv2 Encoder
    Extracts visual embeddings using DINOv2 model
    """
    
    def __init__(self, model_name='facebook/dinov2-vit-base-p14', device=None, feature_type='cls'):
        """
        Args:
            model_name: DINOv2 model name from HuggingFace
            device: Computing device ('cuda' or 'cpu')
            feature_type: Feature extraction method ('cls' for [CLS] token, 'mean' for mean pooling)
        """
        from transformers import AutoModel, AutoImageProcessor
        
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_type = feature_type
        
        # Load model and processor
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, dataloader):
        """
        Extract embeddings from dataloader
        
        Args:
            dataloader: PyTorch DataLoader yielding (images, filenames)
        
        Returns:
            embeddings: Embedding matrix (n_samples, embedding_dim)
            filenames: List of filenames
        """
        all_embeddings = []
        all_filenames = []
        
        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc=f"Extracting DINOv2 embeddings"):
                # DINOv2 requires processor
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                
                if self.feature_type == 'cls':
                    # Use [CLS] token
                    embeddings = last_hidden_state[:, 0, :]
                elif self.feature_type == 'mean':
                    # Mean pooling over all patch tokens
                    embeddings = last_hidden_state[:, 1:, :].mean(dim=1)
                else:
                    raise ValueError(f"Unsupported feature type: {self.feature_type}")
                
                # L2 normalize
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_filenames.extend(filenames)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        return embeddings, all_filenames


class EmbeddingExtractor:
    """
    Generic Embedding Extractor
    Supports any modality with automatic caching
    """
    
    def __init__(self, cache_dir='./cache'):
        """
        Args:
            cache_dir: Directory for caching extracted embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, data_source, model_name, model_type):
        """
        Generate cache file path
        
        Args:
            data_source: Source data path or identifier
            model_name: Model name
            model_type: Model type identifier
        
        Returns:
            Path to cache file
        """
        source_name = Path(data_source).name
        model_safe = model_name.replace('/', '_').replace('-', '_')
        cache_name = f"{source_name}_{model_type}_{model_safe}.npz"
        return self.cache_dir / cache_name
    
    def extract_image_embeddings(
        self,
        image_dir,
        model_type='clip',
        model_name=None,
        batch_size=32,
        num_workers=0,
        force_extract=False,
        image_size=224
    ):
        """
        Extract embeddings from image folder
        
        Args:
            image_dir: Path to image folder
            model_type: Model type ('clip' or 'dino')
            model_name: Model name (optional, has defaults)
            batch_size: Batch size for processing
            num_workers: Number of data loading workers
            force_extract: Force re-extraction (ignore cache)
            image_size: Image resize dimension
        
        Returns:
            dict: {
                'embeddings': Embedding matrix (n_samples, embedding_dim),
                'filenames': List of filenames,
                'model_type': Model type,
                'model_name': Model name
            }
        """
        # Set default models
        if model_name is None:
            if model_type == 'clip':
                model_name = 'ViT-B/32'
            elif model_type == 'dino':
                model_name = 'facebook/dinov2-vit-base-p14'
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # Check cache
        cache_path = self._get_cache_path(image_dir, model_name, model_type)
        
        if cache_path.exists() and not force_extract:
            print(f"✓ Loading cached embeddings: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            return {
                'embeddings': data['embeddings'],
                'filenames': data['filenames'].tolist(),
                'model_type': model_type,
                'model_name': model_name
            }
        
        # Define image transformation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load dataset
        print(f"✓ Loading images: {image_dir}")
        dataset = ImageFolderDataset(image_dir, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Extract embeddings
        if model_type == 'clip':
            encoder = CLIPImageEncoder(model_name=model_name)
        elif model_type == 'dino':
            encoder = DINOV2Encoder(model_name=model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"✓ Extracting embeddings: {model_type} ({model_name})")
        embeddings, filenames = encoder.encode(dataloader)
        
        # Save cache
        print(f"✓ Saving cache: {cache_path}")
        np.savez(
            cache_path,
            embeddings=embeddings,
            filenames=np.array(filenames)
        )
        
        return {
            'embeddings': embeddings,
            'filenames': filenames,
            'model_type': model_type,
            'model_name': model_name
        }
    
    def extract_joint_image_embeddings(
        self,
        image_dir,
        clip_model='ViT-B/32',
        dino_model='facebook/dinov2-vit-base-p14',
        batch_size=32,
        num_workers=0,
        force_extract=False,
        image_size=224
    ):
        """
        Extract and concatenate CLIP and DINO embeddings
        
        Args:
            image_dir: Path to image folder
            clip_model: CLIP model name
            dino_model: DINO model name
            batch_size: Batch size
            num_workers: Number of workers
            force_extract: Force re-extraction
            image_size: Image size
        
        Returns:
            dict: {
                'clip_embeddings': CLIP embeddings,
                'dino_embeddings': DINO embeddings,
                'joint_embeddings': Concatenated embeddings,
                'filenames': List of filenames
            }
        """
        # Extract CLIP embeddings
        clip_result = self.extract_image_embeddings(
            image_dir,
            model_type='clip',
            model_name=clip_model,
            batch_size=batch_size,
            num_workers=num_workers,
            force_extract=force_extract,
            image_size=image_size
        )
        
        # Extract DINO embeddings
        dino_result = self.extract_image_embeddings(
            image_dir,
            model_type='dino',
            model_name=dino_model,
            batch_size=batch_size,
            num_workers=num_workers,
            force_extract=force_extract,
            image_size=image_size
        )
        
        # Concatenate embeddings
        joint_embeddings = np.concatenate([
            clip_result['embeddings'],
            dino_result['embeddings']
        ], axis=1)
        
        return {
            'clip_embeddings': clip_result['embeddings'],
            'dino_embeddings': dino_result['embeddings'],
            'joint_embeddings': joint_embeddings,
            'filenames': clip_result['filenames']
        }
    
    def clear_cache(self, data_source=None, model_type=None):
        """
        Clear cache
        
        Args:
            data_source: Specific data source (optional)
            model_type: Specific model type (optional)
        """
        if data_source is None and model_type is None:
            # Clear all cache
            for cache_file in self.cache_dir.glob('*.npz'):
                cache_file.unlink()
            print(f"✓ Cleared all cache")
        else:
            # Clear specific cache
            source_name = Path(data_source).name if data_source else '*'
            model_pattern = f'*{model_type}*' if model_type else '*'
            
            for cache_file in self.cache_dir.glob(f'{source_name}_{model_pattern}.npz'):
                cache_file.unlink()
                print(f"✓ Cleared: {cache_file}")
