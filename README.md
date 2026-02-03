# Build Your Own GPT: Creating a Custom Text Generation Engine

This repository contains a comprehensive project on building and training custom GPT-like language models for text generation. It includes training a compact GPT-2 model for generating children's stories and an assignment implementing a Python code-focused inference system.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Main Components](#main-components)
- [Contact](#Contact)

## Features

### Tiny LLM Story Generator
- **Custom GPT-2 Training**: Train a small GPT-2 model (256 embedding dimensions, 4 layers, 4 heads) on the TinyStories dataset
- **Streaming Data Processing**: Efficient data loading and preprocessing for large datasets
- **Checkpointing & Resuming**: Save and resume training with periodic checkpoints
- **Text Generation**: Generate short children's stories from prompts like "Once upon a time"
- **Training Visualization**: Track and plot loss curves during training
- **GPU Optimization**: Configured for GPU training with gradient clipping and mixed precision

### Python Code-Focused GPT-2 Inference
- **Filtered Responses**: Only answers Python programming questions using keyword-based filtering
- **Smart Prompt Enhancement**: Enhances user prompts for better coding responses
- **Comprehensive Testing**: Built-in test suite with 16 diverse prompts
- **Error Handling**: Robust input validation and exception handling
- **Performance Metrics**: Tracks generation time, token counts, and response quality

## Requirements

- Python 3.x
- PyTorch (with CUDA support for GPU acceleration)
- Transformers library
- Datasets library
- TQDM, Matplotlib, NumPy
- Google Colab (recommended for GPU access) or local environment with sufficient GPU memory


### Training the Tiny LLM
1. Open `Build_Your_Own_GPT_Creating_a_Custom_Text_Generation_Engine.ipynb`
2. Follow the sections in order:
   - Mount Google Drive
   - Install dependencies and load TinyStories dataset
   - Configure and initialize the model
   - Run the training loop (10 epochs recommended)
   - Generate sample stories from checkpoints

### Python Code Inference
1. Open `Assignment_Python_Code_Focused_GPT2_Inference.ipynb`
2. Load the pre-trained GPT-2 model (gpt2-medium recommended)
3. Initialize the Python coding filter
4. Test with various prompts using the interactive demo
5. Run the comprehensive test suite to validate filtering

### Key Parameters
- **Model Size**: GPT-2 with 256 embeddings, 4 layers, 4 heads
- **Context Length**: 512 tokens
- **Batch Size**: 52 (adjust based on GPU memory)
- **Learning Rate**: 5e-5 with AdamW optimizer
- **Training Samples**: ~2.1M from TinyStories dataset

## Project Structure

```
├── Build_Your_Own_GPT_Creating_a_Custom_Text_Generation_Engine.ipynb  # Main training notebook
├── Assignment_Python_Code_Focused_GPT2_Inference.ipynb               # Python coding assignment
├── GPT-1.pdf                                                          # Reference material
└── README.md                                                          # This file
```

## Main Components

### TinyStoriesStreamDataset
Custom PyTorch dataset class for streaming TinyStories data with:
- Text cleaning and normalization
- Tokenization with GPT-2 tokenizer
- Next-token prediction formatting
- Configurable block size and minimum length filtering

### PythonCodingFilter
Intelligent filtering system that:
- Monitors 50+ Python-related keywords
- Uses regex patterns for question detection
- Provides detailed filtering reasons
- Handles edge cases and invalid inputs

### PythonCodingAssistant
Complete inference system featuring:
- Enhanced prompt engineering for coding questions
- Configurable generation parameters
- Response cleaning and formatting
- Performance tracking and metadata

## 📞 Contact

- **Author**: Rohan paswan
- **Email**:rohanpaswan001782@gmail.com
