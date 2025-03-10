import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

def visualize_prototypes_3d(prototypes, trait_names):
    """
    Visualize 3D projection of psychological trait prototypes.
    
    Args:
        prototypes (torch.Tensor): Hyperspherical prototypes
        trait_names (list): Names of the psychological traits
    """
    # Convert to numpy for visualization
    proto_np = prototypes.detach().cpu().numpy()
    
    # If dimensions > 3, use PCA to reduce to 3D
    if proto_np.shape[1] > 3:
        pca = PCA(n_components=3)
        proto_3d = pca.fit_transform(proto_np)
    else:
        proto_3d = proto_np
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(proto_3d[:, 0], proto_3d[:, 1], proto_3d[:, 2], s=100, alpha=0.8)
    
    # Add labels
    for i, trait in enumerate(trait_names):
        ax.text(proto_3d[i, 0], proto_3d[i, 1], proto_3d[i, 2], trait, size=12)
    
    # Add title and labels
    ax.set_title("3D Visualization of Psychological Trait Prototypes", size=16)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    
    # Add a unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Scale the sphere to match the projection
    max_range = np.max([
        proto_3d[:, 0].max() - proto_3d[:, 0].min(),
        proto_3d[:, 1].max() - proto_3d[:, 1].min(),
        proto_3d[:, 2].max() - proto_3d[:, 2].min()
    ])
    scale = max_range / 2.0
    
    ax.plot_wireframe(x * scale, y * scale, z * scale, color='gray', alpha=0.2)
    
    plt.tight_layout()
    plt.show()

def visualize_prototype_similarities(prototypes, trait_names):
    """
    Visualize similarities between psychological trait prototypes.
    
    Args:
        prototypes (torch.Tensor): Hyperspherical prototypes
        trait_names (list): Names of the psychological traits
    """
    # Compute similarities (cosine distances)
    similarities = torch.mm(prototypes, prototypes.t()).detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarities, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=trait_names, yticklabels=trait_names, vmin=-1, vmax=1)
    plt.title("Similarities Between Psychological Trait Prototypes", size=16)
    plt.tight_layout()
    plt.show()

def visualize_subject_traits_radar(subject_results):
    """
    Create a radar chart of a subject's psychological traits.
    
    Args:
        subject_results (dict): Dictionary mapping trait names to scores
    """
    # Prepare data
    traits = list(subject_results.keys())
    values = list(subject_results.values())
    
    # Number of variables
    N = len(traits)
    
    # What will be the angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Values for the plot, with a loop closure
    values += values[:1]
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw the polygon and fill area
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits)
    
    # Set y limits
    ax.set_ylim(0, 1)
    
    # Add title
    plt.title("Psychological Trait Profile", size=16)
    
    plt.tight_layout()
    plt.show()

def visualize_population_distribution(model, dataset, trait_names, n_samples=1000):
    """
    Visualize the distribution of psychological traits in a population.
    
    Args:
        model (PsycheHPN): Trained model
        dataset: Dataset containing population samples
        trait_names (list): Names of the psychological traits
        n_samples (int): Number of samples to visualize
    """
    # Get device
    device = next(model.parameters()).device
    
    # Get predictions for a subset of the population
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=min(n_samples, len(dataset)), shuffle=True
    )
    
    features, _ = next(iter(dataloader))
    features = features.to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        similarities = model.predict(outputs)
        
        # Normalize to [0, 1]
        trait_scores = (similarities + 1) / 2
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(trait_scores.cpu().numpy(), columns=trait_names)
    
    # Create violin plots for each trait
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df)
    plt.title("Distribution of Psychological Traits in Population", size=16)
    plt.ylabel("Trait Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Create a correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Psychological Traits", size=16)
    plt.tight_layout()
    plt.show()

def visualize_embeddings_tsne(model, dataset, trait_scores, trait_names, n_samples=1000):
    """
    Visualize t-SNE of subject embeddings colored by dominant trait.
    
    Args:
        model (PsycheHPN): Trained model
        dataset: Dataset containing population samples
        trait_scores (torch.Tensor): Ground truth trait scores
        trait_names (list): Names of the psychological traits
        n_samples (int): Number of samples to visualize
    """
    # Get device
    device = next(model.parameters()).device
    
    # Get embeddings for a subset of the population
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=min(n_samples, len(dataset)), shuffle=True
    )
    
    features, _ = next(iter(dataloader))
    features = features.to(device)
    
    # Get model embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(features)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    # Get dominant trait for each subject
    if isinstance(trait_scores, torch.Tensor):
        dominant_traits = trait_scores.argmax(dim=1).cpu().numpy()
    else:
        dominant_traits = np.argmax(trait_scores, axis=1)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=dominant_traits, 
        cmap='tab10', 
        alpha=0.8,
        s=50
    )
    
    # Add legend
    legend1 = plt.legend(
        scatter.legend_elements()[0], 
        [trait_names[i] for i in range(len(trait_names))],
        title="Dominant Trait"
    )
    plt.gca().add_artist(legend1)
    
    plt.title("t-SNE Visualization of Psychological Embeddings", size=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()

def main():
    """Example usage of visualization functions"""
    # Import from psyche_hpn module
    from psyche_hpn import generate_trait_prototypes, psychological_traits_example
    
    # Get trait names
    trait_names = psychological_traits_example()
    num_traits = len(trait_names)
    
    # Generate trait prototypes
    output_dims = 16
    trait_prototypes = generate_trait_prototypes(num_traits, output_dims)
    
    # Visualize prototypes in 3D
    visualize_prototypes_3d(trait_prototypes, trait_names)
    
    # Visualize prototype similarities
    visualize_prototype_similarities(trait_prototypes, trait_names)
    
    # Example subject analysis
    subject_results = {
        trait: np.random.uniform(0.2, 0.8) for trait in trait_names
    }
    
    # Visualize subject traits
    visualize_subject_traits_radar(subject_results)
    
    print("Visualization example complete!")

if __name__ == "__main__":
    main()
