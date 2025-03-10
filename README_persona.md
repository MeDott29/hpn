# Hyperspherical Prototype Networks for Persona Classification

This project implements Hyperspherical Prototype Networks (HPN) for classifying personas from the PersonaHub dataset. The implementation is based on the paper [Hyperspherical Prototype Networks](https://arxiv.org/abs/1901.10514) by Mettes et al. (2019).

## Overview

Hyperspherical Prototype Networks represent classes using prototypes on a hypersphere, which enables efficient classification while maintaining good separation between classes. This project applies this approach to persona classification using the [PersonaHub dataset](https://huggingface.co/datasets/proj-persona/PersonaHub), which contains a large collection of diverse personas.

The main components of this project are:
1. **hpn_persona.py**: Implementation of the Hyperspherical Prototype Network for persona classification
2. **benchmark_persona.py**: Script for benchmarking different configurations of the model
3. **prototypes.py**: Utility for generating hyperspherical prototypes

## Requirements

To run this project, you need the following dependencies:

```
torch>=1.8.0
torchvision
transformers>=4.5.0
datasets>=1.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.60.0
pandas>=1.2.0
numpy>=1.19.0
psutil
```

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

To train a Hyperspherical Prototype Network on the PersonaHub dataset:

```bash
python hpn_persona.py --batch_size 64 --epochs 50 --lr 0.001 --prototype_dim 128 --num_prototypes 100 --num_samples 10000 --visualize --use_amp --pin_memory --num_workers 4
```

Parameters:
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of epochs for training (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--momentum`: Momentum for SGD (default: 0.9)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--embedding_dim`: Dimension of text embeddings (default: 768)
- `--prototype_dim`: Dimension of prototypes (default: 128)
- `--num_prototypes`: Number of prototypes (default: 100)
- `--num_samples`: Number of samples to use from dataset (default: 10000)
- `--seed`: Random seed (default: 42)
- `--data_dir`: Directory to store dataset (default: ./data/personahub)
- `--results_dir`: Directory to store results (default: ./results)
- `--visualize`: Visualize embeddings and prototypes
- `--use_amp`: Use automatic mixed precision training for faster GPU training
- `--num_workers`: Number of workers for data loading (default: 4)
- `--pin_memory`: Pin memory for faster GPU transfer
- `--gpu_id`: GPU ID to use (default: 0)
- `--prefetch_factor`: Prefetch factor for data loading (default: 2)

### Running Benchmarks

To benchmark different configurations of the model:

```bash
python benchmark_persona.py --num_runs 3 --visualize --use_amp --pin_memory --num_workers 4
```

Parameters:
- `--results_dir`: Directory to store benchmark results (default: ./benchmark_results)
- `--num_runs`: Number of runs for each configuration (default: 3)
- `--visualize`: Visualize benchmark results
- `--use_amp`: Use automatic mixed precision training for faster GPU training
- `--num_workers`: Number of workers for data loading (default: 4)
- `--pin_memory`: Pin memory for faster GPU transfer
- `--gpu_id`: GPU ID to use (default: 0)

## GPU Optimization

The implementation includes several optimizations for efficient GPU usage:

1. **Automatic Mixed Precision (AMP)**: Uses mixed precision training to speed up computation and reduce memory usage.
2. **Memory Management**: Periodically clears GPU cache to prevent out-of-memory errors.
3. **Data Loading Optimization**: Uses multiple workers, pin memory, and prefetching for faster data loading.
4. **GPU Monitoring**: Tracks and displays GPU memory usage during training.
5. **Efficient Tensor Transfer**: Uses non-blocking tensor transfers to overlap computation and data transfer.

These optimizations are particularly useful for the NVIDIA GeForce GTX 1060 6GB GPU, which has limited memory (6GB).

## Methodology

The implementation follows these steps:

1. **Data Loading**: The PersonaHub dataset is loaded from Hugging Face and preprocessed.
2. **Text Encoding**: Persona descriptions are encoded using a pretrained BERT model.
3. **Prototype Initialization**: Prototypes are initialized on the unit hypersphere.
4. **Training**: The model is trained to minimize both classification loss and prototype separation loss.
5. **Evaluation**: The model is evaluated on a test set, and visualizations are generated.

## Benchmark Results

The benchmark script evaluates different configurations of the model, varying:
- Prototype dimensions: 64, 128, 256
- Number of prototypes: 50, 100, 200

For each configuration, the script measures:
- Classification accuracy
- Runtime
- Prototype separation
- GPU memory usage

The results are visualized using:
- Line plots of accuracy vs. prototype dimension and number of prototypes
- Heatmap of average accuracy by configuration
- Box plot of accuracy distribution
- Bar plot of runtime by configuration

## Visualizations

The model generates t-SNE visualizations of embeddings and prototypes, which help understand how well the model separates different persona categories in the embedding space.

## References

- Mettes, P., van der Pol, E., & Snoek, C. G. M. (2019). Hyperspherical Prototype Networks. In Advances in Neural Information Processing Systems.
- PersonaHub dataset: [https://huggingface.co/datasets/proj-persona/PersonaHub](https://huggingface.co/datasets/proj-persona/PersonaHub) 