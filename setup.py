"""Setup script for edge-slm-ace package."""

from setuptools import find_packages, setup

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Domain-Specific Benchmarking of Small Language Models for Edge Devices with Agentic Context Engineering"

setup(
    name="edge-slm-ace",
    version="0.1.0",
    description="Domain-Specific Benchmarking of Small Language Models for Edge Devices with Agentic Context Engineering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Suryodaya, Sathwik, Archit",
    author_email="team@example.com",
    url="https://github.com/SirAlchemist1/edge-slm-ace",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "metrics": [
            "sacrebleu>=2.3.0",
            "sentence-transformers>=2.2.0",
            "scikit-learn>=1.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
