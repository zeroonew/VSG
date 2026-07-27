from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vsg",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="VSG: Voxel-based Subspace Grid for dataset coverage and similarity analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vsg",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "transformers>=4.20.0",
        "clip-by-openai>=1.0.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
    ],
)
