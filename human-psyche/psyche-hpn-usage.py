import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from psyche_hpn import PsycheHPN, generate_trait_prototypes, analyze_psyche, train_psyche_hpn
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    """Custom dataset for your own psychological data"""
    
    def __init__(self, data_path, trait_columns, transform=None):
        """
        Initialize with path to data file.
        
        Args:
            data_path (str): Path to CSV file with data
            trait_columns (list): Names of columns containing trait scores
            transform (callable, optional): Optional transform to apply to features
        """
        # Load data from CSV
        self.data = pd.read_csv(data_path)
        
        # Extract features (all columns except trait scores)
        feature_columns = [col for col in self.data.columns if col not in trait_columns]
        self.features = self.data[feature_columns].values
        
        # Extract trait scores
        self.trait_scores = self.data[trait_columns].values
        
        # Normalize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)
        
        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.trait_scores = torch.tensor(self.trait_scores, dtype=torch.float32)
        
        # Save transform function
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        trait_scores = self.trait_scores[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, trait_scores

def analyze_text_data(model, text, tokenizer, trait_names, max_length=128):
    """
    Analyze psychological traits from text data using a trained model.
    
    Args:
        model (PsycheHPN): Trained model
        text (str): Text to analyze
        tokenizer: Text tokenizer (e.g., from transformers library)
        trait_names (list): Names of psychological traits
        max_length (int): Maximum token length
        
    Returns:
        dict: Dictionary mapping trait names to scores
    """
    # Tokenize text
    tokens = tokenizer(text, padding='max_length', max_length=max_length, 
                      truncation=True, return_tensors='pt')
    
    # Extract input features (e.g., using [CLS] token or average of all tokens)
    with torch.no_grad():
        # For BERT-like models, use the [CLS] token
        features = tokens['input_ids'].unsqueeze(0)
        
        # Analyze with model
        results = analyze_psyche(model, features, trait_names)
    
    return results

def main_custom_data():
    """Example of using the PsycheHPN with your own data"""
    
    # Define paths and parameters
    data_path = "your_psychological_data.csv"  # Replace with your data path
    trait_columns = [
        "openness", 
        "conscientiousness", 
        "extraversion", 
        "agreeableness", 
        "neuroticism"
    ]  # Replace with your trait column names
    
    # Model parameters
    input_size = 64       # Number of input features
    hidden_size = 128     # Size of hidden layers
    output_dims = 32      # Dimensionality of hypersphere
    num_traits = len(trait_columns)
    batch_size = 32
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    try:
        dataset = CustomDataset(data_path, trait_columns)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print(f"Could not find data file at {data_path}")
        print("Using synthetic data for demonstration...")
        
        # Generate synthetic data if real data is not available
        num_samples = 1000
        features = torch.randn(num_samples, input_size)
        trait_scores = torch.rand(num_samples, num_traits)
        
        # Create simple dataset
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(features, trait_scores)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Generate trait prototypes
    trait_prototypes = generate_trait_prototypes(num_traits, output_dims)
    
    # Create model
    model = PsycheHPN(input_size, hidden_size, output_dims, trait_prototypes)
    
    # Train model
    trained_model = train_psyche_hpn(model, train_loader, epochs=epochs, device=device)
    
    # Save trained model
    torch.save(trained_model.state_dict(), "psyche_hpn_model.pth")
    print("Model saved to psyche_hpn_model.pth")
    
    # Example: Analyze a new person's data
    new_data = torch.randn(1, input_size)  # Replace with actual data
    results = analyze_psyche(trained_model, new_data, trait_columns)
    
    # Print results
    print("\nPsychological Trait Analysis:")
    for trait, score in results.items():
        print(f"{trait}: {score:.4f}")

def load_pretrained_model(model_path, input_size, hidden_size, output_dims, trait_prototypes):
    """Load a pretrained PsycheHPN model"""
    model = PsycheHPN(input_size, hidden_size, output_dims, trait_prototypes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

if __name__ == "__main__":
    main_custom_data()
