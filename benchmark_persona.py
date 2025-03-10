#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for evaluating Hyperspherical Prototype Networks on PersonaHub dataset.

This script runs a series of experiments with different configurations to benchmark
the performance of hyperspheric prototype networks on persona classification.
"""

import os
import sys
import numpy as np
import json
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import time
import subprocess
from itertools import product

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Hyperspherical Prototype Networks on PersonaHub")
    parser.add_argument('--results_dir', type=str, default='./benchmark_results', help='Directory to store benchmark results')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs for each configuration')
    parser.add_argument('--visualize', action='store_true', help='Visualize benchmark results')
    parser.add_argument('--use_amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--pin_memory', action='store_true', help='Pin memory for faster GPU transfer')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()
    return args

def run_experiment(config, results_dir, run_idx):
    """Run a single experiment with the given configuration."""
    
    # Create command
    cmd = [
        "python", "hpn_persona.py",
        "--batch_size", str(config["batch_size"]),
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"]),
        "--momentum", str(config["momentum"]),
        "--weight_decay", str(config["weight_decay"]),
        "--prototype_dim", str(config["prototype_dim"]),
        "--num_prototypes", str(config["num_prototypes"]),
        "--num_samples", str(config["num_samples"]),
        "--seed", str(config["seed"] + run_idx),  # Different seed for each run
        "--data_dir", f"{results_dir}/data",
        "--results_dir", f"{results_dir}/run_{run_idx}",
        "--num_workers", str(config["num_workers"]),
        "--gpu_id", str(config["gpu_id"])
    ]
    
    if config.get("visualize", False):
        cmd.append("--visualize")
    
    if config.get("use_amp", False):
        cmd.append("--use_amp")
    
    if config.get("pin_memory", False):
        cmd.append("--pin_memory")
    
    # Run the experiment
    print(f"Running experiment with config: {config}, run: {run_idx+1}")
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Check for errors
    if process.returncode != 0:
        print(f"Error running experiment: {stderr}")
        return None
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Load results
    try:
        with open(f"{results_dir}/run_{run_idx}/results.json", "r") as f:
            results = json.load(f)
        
        # Add runtime and configuration to results
        results["runtime"] = runtime
        results.update(config)
        results["run_idx"] = run_idx
        
        return results
    except FileNotFoundError:
        print(f"Results file not found for run {run_idx}")
        return None

def generate_configurations(args):
    """Generate different configurations for benchmarking."""
    
    # Base configuration
    base_config = {
        "batch_size": 64,
        "epochs": 30,
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "seed": 42,
        "num_samples": 10000,
        "visualize": True,
        "use_amp": args.use_amp,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "gpu_id": args.gpu_id
    }
    
    # Parameters to vary
    prototype_dims = [64, 128, 256]
    num_prototypes = [50, 100, 200]
    
    # Generate all combinations
    configs = []
    for dim, num in product(prototype_dims, num_prototypes):
        config = base_config.copy()
        config["prototype_dim"] = dim
        config["num_prototypes"] = num
        configs.append(config)
    
    return configs

def visualize_results(results_df, results_dir):
    """Visualize benchmark results."""
    
    # Create directory for visualizations
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Accuracy vs. Prototype Dimension
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="prototype_dim", y="accuracy", hue="num_prototypes", marker="o")
    plt.title("Accuracy vs. Prototype Dimension")
    plt.xlabel("Prototype Dimension")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(vis_dir, "accuracy_vs_dim.png"))
    plt.close()
    
    # 2. Accuracy vs. Number of Prototypes
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="num_prototypes", y="accuracy", hue="prototype_dim", marker="o")
    plt.title("Accuracy vs. Number of Prototypes")
    plt.xlabel("Number of Prototypes")
    plt.ylabel("Accuracy (%)")
    plt.savefig(os.path.join(vis_dir, "accuracy_vs_num_prototypes.png"))
    plt.close()
    
    # 3. Runtime vs. Configuration
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x="config_name", y="runtime")
    plt.title("Runtime vs. Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Runtime (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "runtime_vs_config.png"))
    plt.close()
    
    # 4. Heatmap of average accuracy
    pivot_df = results_df.pivot_table(
        values="accuracy", 
        index="prototype_dim", 
        columns="num_prototypes", 
        aggfunc="mean"
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Average Accuracy by Configuration")
    plt.xlabel("Number of Prototypes")
    plt.ylabel("Prototype Dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "accuracy_heatmap.png"))
    plt.close()
    
    # 5. Box plot of accuracy distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x="config_name", y="accuracy")
    plt.title("Accuracy Distribution by Configuration")
    plt.xlabel("Configuration")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "accuracy_boxplot.png"))
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}")

def main():
    args = parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Generate configurations
    configs = generate_configurations(args)
    
    # Run experiments
    all_results = []
    
    for config_idx, config in enumerate(configs):
        config_name = f"dim{config['prototype_dim']}_proto{config['num_prototypes']}"
        config_dir = os.path.join(args.results_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        
        # Save configuration
        with open(os.path.join(config_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
        
        # Run multiple times with different seeds
        config_results = []
        for run_idx in range(args.num_runs):
            result = run_experiment(config, config_dir, run_idx)
            if result:
                result["config_name"] = config_name
                config_results.append(result)
                all_results.append(result)
        
        # Compute average results for this configuration
        if config_results:
            avg_accuracy = np.mean([r["accuracy"] for r in config_results])
            avg_runtime = np.mean([r["runtime"] for r in config_results])
            
            print(f"Configuration {config_name}:")
            print(f"  Average Accuracy: {avg_accuracy:.2f}%")
            print(f"  Average Runtime: {avg_runtime:.2f} seconds")
            print()
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(args.results_dir, "all_results.csv"), index=False)
    
    # Generate summary
    summary = results_df.groupby("config_name").agg({
        "accuracy": ["mean", "std", "min", "max"],
        "runtime": ["mean", "std", "min", "max"],
        "prototype_dim": "first",
        "num_prototypes": "first",
        "use_amp": "first"
    }).reset_index()
    
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    
    # Visualize results
    if args.visualize and len(all_results) > 0:
        visualize_results(results_df, args.results_dir)
    
    print(f"Benchmark completed. Results saved to {args.results_dir}")

if __name__ == "__main__":
    main() 