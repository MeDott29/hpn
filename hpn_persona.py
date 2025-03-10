#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperspherical Prototype Networks for Persona Classification

This script implements a hyperspheric prototype network for classifying personas
from the PersonaHub dataset. It downloads the dataset from Hugging Face,
processes it, and trains a prototype-based classifier with hyperspheric embeddings.

@inproceedings{mettes2016hyperspherical,
 title={Hyperspherical Prototype Networks},
 author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
 booktitle={Advances in Neural Information Processing Systems},
 year={2019}
}
"""

import os
import sys
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import re
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import helper
import gc
import time
from torch.cuda.amp import autocast, GradScaler
import psutil

# Parse user arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical Prototype Networks for Persona Classification")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Dimension of text embeddings')
    parser.add_argument('--prototype_dim', type=int, default=128, help='Dimension of prototypes')
    parser.add_argument('--num_prototypes', type=int, default=100, help='Number of prototypes')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to use from dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data/personahub', help='Directory to store dataset')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to store results')
    parser.add_argument('--visualize', action='store_true', help='Visualize embeddings and prototypes')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster GPU transfer')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor for data loading')
    args = parser.parse_args()
    return args

# PersonaHub Dataset class
class PersonaHubDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Text Encoder model
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model, embedding_dim, prototype_dim):
        super(TextEncoder, self).__init__()
        self.encoder = pretrained_model
        self.projection = nn.Linear(embedding_dim, prototype_dim)
        
    def forward(self, input_ids, attention_mask):
        # Get the embeddings from the encoder
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        
        # Project to prototype space and normalize
        projected = self.projection(embeddings)
        normalized = F.normalize(projected, p=2, dim=1)
        
        return normalized

# Hyperspherical Prototype Network
class HypersphericalProtoNet(nn.Module):
    def __init__(self, encoder, num_prototypes, prototype_dim):
        super(HypersphericalProtoNet, self).__init__()
        self.encoder = encoder
        
        # Initialize prototypes on the unit hypersphere
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, prototype_dim))
        self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=1)
        
    def forward(self, input_ids, attention_mask):
        # Get normalized embeddings
        embeddings = self.encoder(input_ids, attention_mask)
        
        # Normalize prototypes
        normalized_prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # Compute cosine similarities (dot product of normalized vectors)
        similarities = torch.matmul(embeddings, normalized_prototypes.t())
        
        # Scale similarities for softmax
        similarities = similarities * 10.0  # Temperature scaling
        
        return similarities, embeddings, normalized_prototypes

# Compute the prototype separation loss
def prototype_loss(prototypes):
    # Dot product of normalized prototypes is cosine similarity
    product = torch.matmul(prototypes, prototypes.t())
    
    # Remove diagonal from loss
    mask = 1.0 - torch.eye(prototypes.size(0), device=prototypes.device)
    product = product * mask
    
    # Minimize maximum cosine similarity
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()

# Load and preprocess the PersonaHub dataset
def load_personahub_data(args):
    print("Loading PersonaHub dataset...")
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("proj-persona/PersonaHub", "persona")
    
    # Extract persona descriptions and categories
    personas = []
    categories = []
    
    # Process the dataset
    for item in dataset['train']:
        persona_desc = item['persona']
        # Extract main category from persona description
        # This is a simplification - in a real implementation, you might want to use more sophisticated categorization
        category = extract_category(persona_desc)
        
        personas.append(persona_desc)
        categories.append(category)
    
    # Limit the number of samples if specified
    if args.num_samples > 0 and args.num_samples < len(personas):
        personas = personas[:args.num_samples]
        categories = categories[:args.num_samples]
    
    # Encode categories to numerical labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categories)
    
    # Get the mapping from numerical labels to category names
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
    # Split the data into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        personas, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    
    print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
    print(f"Number of categories: {len(label_mapping)}")
    
    return train_texts, test_texts, train_labels, test_labels, label_mapping

# Extract a category from persona description
def extract_category(persona_desc):
    # This is a simplified approach - in a real implementation, you might want to use more sophisticated categorization
    # For example, you could use topic modeling or clustering
    
    # Look for profession or role indicators
    profession_patterns = [
        r'(professor|teacher|educator|instructor)',
        r'(doctor|physician|surgeon|nurse|healthcare)',
        r'(engineer|developer|programmer|coder)',
        r'(artist|painter|musician|composer|creative)',
        r'(writer|author|poet|journalist)',
        r'(scientist|researcher|analyst)',
        r'(business|entrepreneur|executive|manager)',
        r'(lawyer|attorney|legal)',
        r'(student|learner|scholar)',
        r'(parent|mother|father|caregiver)'
    ]
    
    for i, pattern in enumerate(profession_patterns):
        if re.search(pattern, persona_desc.lower()):
            categories = ['educator', 'healthcare', 'tech', 'creative', 'writer', 
                         'scientist', 'business', 'legal', 'student', 'parent']
            return categories[i]
    
    # Default category if no match is found
    return 'other'

# Print GPU memory usage
def print_gpu_memory_usage(device):
    if torch.cuda.is_available():
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  Cached:    {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"  Free:      {torch.cuda.get_device_properties(device).total_memory / 1024**2 - torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

# Train the model
def train(model, train_loader, optimizer, device, epoch, use_amp=False, scaler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if use_amp:
            with autocast():
                logits, embeddings, prototypes = model(input_ids, attention_mask)
                
                # Classification loss
                classification_loss = F.cross_entropy(logits, labels)
                
                # Prototype separation loss
                proto_loss, max_sim = prototype_loss(prototypes)
                
                # Total loss
                loss = classification_loss + 0.1 * proto_loss
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            logits, embeddings, prototypes = model(input_ids, attention_mask)
            
            # Classification loss
            classification_loss = F.cross_entropy(logits, labels)
            
            # Prototype separation loss
            proto_loss, max_sim = prototype_loss(prototypes)
            
            # Total loss
            loss = classification_loss + 0.1 * proto_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Renormalize prototypes
        with torch.no_grad():
            model.prototypes.data = F.normalize(model.prototypes.data, p=2, dim=1)
        
        # Update statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'proto_sim': max_sim.item(),
            'gpu_mem': f"{torch.cuda.memory_allocated(device) / 1024**2:.0f}MB"
        })
        
        # Periodically clear cache to prevent OOM
        if batch_idx % 50 == 0 and batch_idx > 0:
            torch.cuda.empty_cache()
    
    return total_loss / len(train_loader), 100. * correct / total

# Evaluate the model
def evaluate(model, test_loader, device, use_amp=False):
    model.eval()
    correct = 0
    total = 0
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision if enabled
            if use_amp:
                with autocast():
                    logits, embeddings, _ = model(input_ids, attention_mask)
            else:
                logits, embeddings, _ = model(input_ids, attention_mask)
            
            # Compute accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store embeddings and labels for visualization
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return accuracy, np.vstack(all_embeddings), np.concatenate(all_labels)

# Visualize embeddings and prototypes
def visualize_embeddings(embeddings, labels, prototypes, label_mapping, save_path):
    print("Visualizing embeddings and prototypes...")
    
    # Use a subset of embeddings for visualization if there are too many
    max_samples = 5000
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Combine embeddings and prototypes
    combined = np.vstack([embeddings, prototypes.detach().cpu().numpy()])
    
    # Create labels for the combined data
    combined_labels = np.concatenate([labels, np.arange(len(prototypes)) + len(np.unique(labels))])
    
    # Apply t-SNE for dimensionality reduction
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1))
    reduced = tsne.fit_transform(combined)
    
    # Split back into embeddings and prototypes
    reduced_embeddings = reduced[:len(embeddings)]
    reduced_prototypes = reduced[len(embeddings):]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot embeddings
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(
            reduced_embeddings[idx, 0],
            reduced_embeddings[idx, 1],
            alpha=0.5,
            label=f"{label_mapping[label]}"
        )
    
    # Plot prototypes
    plt.scatter(
        reduced_prototypes[:, 0],
        reduced_prototypes[:, 1],
        c='black',
        marker='*',
        s=200,
        label='Prototypes'
    )
    
    plt.title('t-SNE Visualization of Embeddings and Prototypes')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Visualization saved to {save_path}")

# Main function
def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = True  # For reproducibility
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        print_gpu_memory_usage(device)
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU instead.")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Initialize mixed precision scaler if using AMP
    scaler = GradScaler() if args.use_amp else None
    
    # Load and preprocess data
    train_texts, test_texts, train_labels, test_labels, label_mapping = load_personahub_data(args)
    
    # Initialize tokenizer and model
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Clear GPU memory before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Load pretrained model
    pretrained_model = AutoModel.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = PersonaHubDataset(train_texts, train_labels, tokenizer)
    test_dataset = PersonaHubDataset(test_texts, test_labels, tokenizer)
    
    # Create data loaders with optimized settings for GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0
    )
    
    # Initialize model
    encoder = TextEncoder(pretrained_model, args.embedding_dim, args.prototype_dim)
    model = HypersphericalProtoNet(encoder, args.num_prototypes, args.prototype_dim)
    model = model.to(device)
    
    # Print model summary
    print(f"Model parameters: {helper.count_parameters(model)}")
    
    # Initialize optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    best_accuracy = 0
    start_time = time.time()
    
    print(f"Starting training with {'mixed precision' if args.use_amp else 'full precision'}...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, device, epoch, args.use_amp, scaler)
        
        # Evaluate the model
        test_acc, embeddings, labels = evaluate(model, test_loader, device, args.use_amp)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        # Print GPU memory usage
        print_gpu_memory_usage(device)
        
        # Save the best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), os.path.join(args.results_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
            
            # Visualize embeddings and prototypes
            if args.visualize:
                visualize_embeddings(
                    embeddings,
                    labels,
                    model.prototypes,
                    label_mapping,
                    os.path.join(args.results_dir, 'embeddings_visualization.png')
                )
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    
    # Final evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.results_dir, 'best_model.pth')))
    final_acc, final_embeddings, final_labels = evaluate(model, test_loader, device, args.use_amp)
    
    # Save results
    results = {
        'accuracy': final_acc,
        'num_prototypes': args.num_prototypes,
        'prototype_dim': args.prototype_dim,
        'num_samples': len(train_texts) + len(test_texts),
        'num_categories': len(label_mapping),
        'categories': label_mapping,
        'training_time': total_time,
        'use_amp': args.use_amp,
        'batch_size': args.batch_size,
        'gpu': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'
    }
    
    with open(os.path.join(args.results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 