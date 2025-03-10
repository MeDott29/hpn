import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class PsycheHPN(nn.Module):
    """
    Hyperspherical Prototype Network for modeling the human psyche.
    Maps arbitrary input data to psychological trait prototypes on a hypersphere.
    """
    
    def __init__(self, input_size, hidden_size, output_dims, trait_prototypes):
        """
        Initialize the network with predefined prototypes for psychological traits.
        
        Args:
            input_size (int): Dimensionality of the input data
            hidden_size (int): Size of the hidden layers
            output_dims (int): Dimensionality of the output space (hypersphere)
            trait_prototypes (torch.Tensor): Predefined prototypes on the hypersphere
        """
        super(PsycheHPN, self).__init__()
        
        # Store trait prototypes
        self.prototypes = trait_prototypes
        
        # Define network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dims)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        """
        Map network output to trait probabilities via cosine similarity
        with the trait prototypes.
        """
        # Normalize output to the hypersphere
        x = F.normalize(x, p=2, dim=1)
        
        # Compute cosine similarity with all prototypes
        similarities = torch.mm(x, self.prototypes.t())
        
        return similarities

def generate_trait_prototypes(num_traits, output_dims, seed=42):
    """
    Generate uniformly distributed prototypes on a hypersphere for 
    psychological traits.
    
    Args:
        num_traits (int): Number of psychological traits to model
        output_dims (int): Dimensionality of the hypersphere
        seed (int): Random seed for reproducibility
        
    Returns:
        torch.Tensor: Normalized trait prototypes on the hypersphere
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize random prototypes
    prototypes = torch.randn(num_traits, output_dims)
    
    # Optimize for uniform distribution through repulsion
    prototypes_param = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
    optimizer = optim.SGD([prototypes_param], lr=0.1, momentum=0.9)
    
    # Optimize for maximum separation
    for i in range(1000):
        # Compute pairwise similarities
        similarities = torch.mm(prototypes_param, prototypes_param.t())
        
        # Remove self-similarities from diagonal
        similarities -= 2.0 * torch.eye(num_traits, device=similarities.device)
        
        # Minimize maximum similarity (maximize separation)
        loss = similarities.max(dim=1)[0].mean()
        
        # Update prototypes
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Re-normalize after gradient step
        with torch.no_grad():
            prototypes_param.data = F.normalize(prototypes_param.data, p=2, dim=1)
    
    return prototypes_param.data

class PsycheDataset(Dataset):
    """Dataset class for psychological trait modeling"""
    
    def __init__(self, features, trait_scores):
        """
        Initialize dataset with features and trait scores.
        
        Args:
            features (torch.Tensor): Input features/data
            trait_scores (torch.Tensor): Target trait scores
        """
        self.features = features
        self.trait_scores = trait_scores
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.trait_scores[idx]

def train_psyche_hpn(model, train_loader, epochs=100, lr=0.001, device='cuda'):
    """
    Train the Psyche HPN model.
    
    Args:
        model (PsycheHPN): The model to train
        train_loader (DataLoader): DataLoader with training data
        epochs (int): Number of epochs to train
        lr (float): Learning rate
        device (str): Device to train on ('cuda' or 'cpu')
    
    Returns:
        model (PsycheHPN): The trained model
    """
    # Move model to device
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Cosine similarity loss to match output to correct prototype
    loss_fn = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for features, targets in train_loader:
            # Move data to device
            features = features.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(features)
            outputs = F.normalize(outputs, p=2, dim=1)
            
            # Convert targets to one-hot encoding of prototype indices
            target_prototypes = model.prototypes[targets.argmax(dim=1)]
            
            # Compute loss: 1 - cosine similarity
            loss = (1 - loss_fn(outputs, target_prototypes)).pow(2).sum()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch statistics
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
    
    return model

def psychological_traits_example():
    """
    Example of psychological traits we might model.
    These could be the Big Five personality traits or other psychological constructs.
    """
    traits = [
        "Openness",          # Openness to experience
        "Conscientiousness", # Organization and reliability
        "Extraversion",      # Sociability and assertiveness
        "Agreeableness",     # Compassion and cooperation
        "Neuroticism",       # Emotional instability
        "Creativity",        # Imaginative thinking
        "Resilience",        # Ability to cope with stress
        "Empathy",           # Understanding others' emotions
    ]
    return traits

def analyze_psyche(model, features, trait_names):
    """
    Analyze psychological traits in input data.
    
    Args:
        model (PsycheHPN): Trained HPN model
        features (torch.Tensor): Input features to analyze
        trait_names (list): List of trait names corresponding to prototypes
    
    Returns:
        dict: Dictionary mapping trait names to strength scores
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get device
    device = next(model.parameters()).device
    
    # Move features to device
    features = features.to(device)
    
    # Get predictions
    with torch.no_grad():
        # Forward pass
        output = model(features)
        
        # Get similarities to prototypes
        similarities = model.predict(output)
        
        # Normalize similarities to [0, 1] range
        normalized_scores = (similarities + 1) / 2
    
    # Create trait-to-score mapping
    results = {}
    for i, trait in enumerate(trait_names):
        results[trait] = normalized_scores[0, i].item()
    
    return results

def main():
    # Define parameters
    input_size = 64       # Size of input features
    hidden_size = 128     # Size of hidden layers
    output_dims = 16      # Dimensionality of hypersphere
    num_traits = 8        # Number of psychological traits to model
    batch_size = 32
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get trait names
    trait_names = psychological_traits_example()
    
    # Generate trait prototypes
    trait_prototypes = generate_trait_prototypes(num_traits, output_dims)
    
    # Create model
    model = PsycheHPN(input_size, hidden_size, output_dims, trait_prototypes)
    
    # Example: Generate synthetic data
    # In a real application, this would be replaced with actual data
    num_samples = 1000
    features = torch.randn(num_samples, input_size)
    
    # Generate synthetic trait scores (one-hot encoding for this example)
    trait_scores = torch.zeros(num_samples, num_traits)
    for i in range(num_samples):
        # Randomly assign a dominant trait
        dominant_trait = np.random.randint(0, num_traits)
        trait_scores[i, dominant_trait] = 1.0
    
    # Create dataset and data loader
    dataset = PsycheDataset(features, trait_scores)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train model
    trained_model = train_psyche_hpn(model, train_loader, epochs=epochs, device=device)
    
    # Example: Analyze new data
    new_data = torch.randn(1, input_size)
    results = analyze_psyche(trained_model, new_data, trait_names)
    
    # Print results
    print("\nPsychological Trait Analysis:")
    for trait, score in results.items():
        print(f"{trait}: {score:.4f}")

if __name__ == "__main__":
    main()
