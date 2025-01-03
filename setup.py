from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="safecall",
    version="0.1.0",
    author="Saksham Hooda",
    author_email="sakshamhooda_mc20a7_62@dtu.ac.in",
    description="AI-Powered Spoof Call Detection System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sakshamhooda/SpoofCall-Detection-1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.2",
        "scipy>=1.7.0",
        "librosa>=0.8.1",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "scikit-learn>=0.24.2",
        "opencv-python>=4.5.3",
        "PyYAML>=5.4.1",
        "tqdm>=4.62.3",
        "pandas>=1.3.3",
        "pillow>=8.3.2",
        "soundfile>=0.10.3",
        "tensorboard>=2.6.0",
        "einops>=0.3.2",
        "timm>=0.4.12",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.9b0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "safecall-train=src.train:main",
            "safecall-infer=src.inference:main",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/sakshamhooda/SpoofCall-Detection-1/issues",
    },
)