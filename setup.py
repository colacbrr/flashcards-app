#!/usr/bin/env python3
"""
Setup script for FlashCard Studio
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="flashcard-studio",
    version="2.0.0",
    description="An intelligent flashcard generator powered by local LLM models",
    long_description=README,
    long_description_content_type="text/markdown",
    author="FlashCard Studio Team",
    author_email="contact@flashcardstudio.dev",
    url="https://github.com/yourusername/flashcard-studio",
    packages=find_packages(),
    py_modules=["flashcard_app"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="flashcards, education, ai, llm, study, learning",
    python_requires=">=3.8",
    install_requires=[
        "PyQt5>=5.15.0",
        "ollama>=0.2.0",
        "pymupdf>=1.23.0",
        "python-docx>=1.1.0",
        "aiohttp>=3.9.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-qt>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
        ],
        "build": [
            "pyinstaller>=6.0.0",
            "wheel>=0.38.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "flashcard-studio=flashcard_app:main",
        ],
        "gui_scripts": [
            "flashcard-studio-gui=flashcard_app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)