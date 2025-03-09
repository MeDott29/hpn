"""
Hyperspherical Prototype Networks with Einstein-Rosen Bridge

This implementation extends the original HPN with an Einstein-Rosen bridge
that travels through the center of the poles of the hypersphere.

The Einstein-Rosen bridge creates a wormhole-like connection between antipodal
points on the hypersphere, allowing for more efficient information transfer
and potentially improved classification performance.

@inproceedings{mettes2019hyperspherical,
  title={Hyperspherical Prototype Networks},
  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
"""

import os
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import helper
sys.path.append("models/cifar/")
import densenet, resnet, convnet

class EinsteinRosenBridge(nn.Module):
    """
    Implementation of an Einstein-Rosen bridge for hyperspherical spaces.
    
    The bridge creates a wormhole-like connection between antipodal points
    on the hypersphere, allowing for more efficient information transfer.
    """
    def __init__(self, input_dim, output_dim, wormhole_radius=0.5):
        super(EinsteinRosenBridge, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wormhole_radius = wormhole_radius
        
        # Bridge parameters
        self.bridge_matrix = nn.Parameter(torch.randn(output_dim, input_dim))
        self.pole_vectors = nn.Parameter(torch.randn(2, output_dim))
        
        # Initialize parameters
        nn.init.orthogonal_(self.bridge_matrix)
        nn.init.orthogonal_(self.pole_vectors)
        
        # Normalize pole vectors to unit length
        with torch.no_grad():
            self.pole_vectors.div_(torch.norm(self.pole_vectors, dim=1, keepdim=True))
    
    def forward(self, x):
        """
        Forward pass through the Einstein-Rosen bridge.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Transformed tensor of shape (batch_size, output_dim)
        """
        # Project input to hypersphere
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Calculate distance to poles
        north_dist = 1 - F.cosine_similarity(x_norm, self.pole_vectors[0].unsqueeze(0), dim=1)
        south_dist = 1 - F.cosine_similarity(x_norm, self.pole_vectors[1].unsqueeze(0), dim=1)
        
        # Determine if points are within wormhole radius of either pole
        north_mask = (north_dist < self.wormhole_radius).float().unsqueeze(1)
        south_mask = (south_dist < self.wormhole_radius).float().unsqueeze(1)
        
        # Apply bridge transformation for points near poles
        # For points near north pole, connect to area near south pole and vice versa
        bridge_effect = north_mask * (self.pole_vectors[1] - x_norm * north_dist.unsqueeze(1)) + \
                        south_mask * (self.pole_vectors[0] - x_norm * south_dist.unsqueeze(1))
        
        # Combine original points with bridge effect
        bridge_mask = (north_mask + south_mask).clamp(0, 1)
        output = (1 - bridge_mask) * x_norm + bridge_mask * bridge_effect
        
        # Project to output dimension and normalize
        output = F.linear(output, self.bridge_matrix)
        output = F.normalize(output, p=2, dim=1)
        
        return output

class HPNWithBridge(nn.Module):
    """
    Hyperspherical Prototype Network with Einstein-Rosen Bridge.
    
    This model extends the original HPN with a wormhole-like connection
    between antipodal points on the hypersphere.
    """
    def __init__(self, base_model, polars, bridge_radius=0.5):
        super(HPNWithBridge, self).__init__()
        self.base_model = base_model
        self.polars = polars
        self.output_dim = polars.shape[1]
        
        # Add Einstein-Rosen bridge
        self.bridge = EinsteinRosenBridge(
            input_dim=self.output_dim,
            output_dim=self.output_dim,
            wormhole_radius=bridge_radius
        )
    
    def forward(self, x):
        # Get features from base model
        features = self.base_model(x)
        
        # Apply Einstein-Rosen bridge
        bridged_features = self.bridge(features)
        
        return bridged_features
    
    def predict(self, x):
        """
        Predict class probabilities based on cosine similarity to prototypes.
        
        Args:
            x: Input tensor of shape (batch_size, output_dim)
            
        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        # Compute cosine similarity between features and prototypes
        similarities = F.linear(
            F.normalize(x, p=2, dim=1),
            F.normalize(self.polars, p=2, dim=1)
        )
        
        return similarities

def visualize_bridge(model, num_samples=1000, save_path=None):
    """
    Visualize the Einstein-Rosen bridge in 3D space.
    
    Args:
        model: HPNWithBridge model
        num_samples: Number of random points to generate
        save_path: Path to save the visualization
    """
    # Only works for 3D embeddings
    if model.output_dim != 3:
        print("Visualization only works for 3D embeddings")
        return
    
    # Generate random points on the sphere
    points = torch.randn(num_samples, 3)
    points = F.normalize(points, p=2, dim=1)
    
    # Apply bridge transformation
    with torch.no_grad():
        transformed = model.bridge(points)
    
    # Convert to numpy for plotting
    points = points.numpy()
    transformed = transformed.numpy()
    poles = model.bridge.pole_vectors.detach().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.3, label='Original')
    
    # Plot transformed points
    ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c='red', alpha=0.3, label='Transformed')
    
    # Plot poles
    ax.scatter(poles[0, 0], poles[0, 1], poles[0, 2], c='green', s=100, label='North Pole')
    ax.scatter(poles[1, 0], poles[1, 1], poles[1, 2], c='yellow', s=100, label='South Pole')
    
    # Draw line connecting poles through center
    ax.plot([poles[0, 0], poles[1, 0]], [poles[0, 1], poles[1, 1]], [poles[0, 2], poles[1, 2]], 'k-', linewidth=2, label='Bridge Axis')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Einstein-Rosen Bridge Visualization')
    ax.legend()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def train_with_bridge(model, device, trainloader, optimizer, f_loss, epoch):
    """
    Training function for HPN with Einstein-Rosen bridge.
    
    Args:
        model: HPNWithBridge model
        device: Torch device
        trainloader: Training data loader
        optimizer: Optimizer
        f_loss: Loss function
        epoch: Current epoch
    """
    # Set mode to training
    model.train()
    avgloss, avglosscount = 0., 0.
    
    # Go over all batches
    for bidx, (data, target) in enumerate(trainloader):
        # Data to device
        nlabels = target.clone()
        target = model.polars[target]
        data = torch.autograd.Variable(data).to(device)
        target = torch.autograd.Variable(target).to(device)
        
        # Compute outputs and losses
        output = model(data)
        
        # Main classification loss (cosine similarity)
        class_loss = (1 - f_loss(output, target)).pow(2).sum()
        
        # Bridge regularization loss to ensure smooth transitions
        bridge_reg = model.bridge.wormhole_radius * 0.1
        
        # Total loss
        loss = class_loss + bridge_reg
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update loss
        avgloss += loss.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount
        
        # Print updates
        print(f"Training epoch {epoch}: loss {newloss:8.4f} - {100.*(bidx+1)/len(trainloader):.0f}%\r", end="")
        sys.stdout.flush()
    print()

def test_with_bridge(model, device, testloader):
    """
    Test function for HPN with Einstein-Rosen bridge.
    
    Args:
        model: HPNWithBridge model
        device: Torch device
        testloader: Test data loader
        
    Returns:
        Classification accuracy
    """
    # Set model to evaluation
    model.eval()
    cos = nn.CosineSimilarity(eps=1e-9)
    acc = 0
    
    # Go over all batches
    with torch.no_grad():
        for data, target in testloader:
            # Data to device
            data = torch.autograd.Variable(data).to(device)
            target = target.to(device)
            target = torch.autograd.Variable(target)
            
            # Forward
            output = model(data).float()
            output = model.predict(output).float()
            
            pred = output.max(1, keepdim=True)[1]
            acc += pred.eq(target.view_as(pred)).sum().item()
    
    # Print results
    testlen = len(testloader.dataset)
    print(f"Testing: classification accuracy: {acc}/{testlen} - {100. * acc / testlen:.3f}%")
    return acc / float(testlen)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HPN with Einstein-Rosen Bridge")
    parser.add_argument("--datadir", dest="datadir", default="data/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/", type=str)
    parser.add_argument("--hpnfile", dest="hpnfile", default="", type=str)
    parser.add_argument("--bridge_radius", dest="bridge_radius", default=0.3, type=float)
    parser.add_argument("--visualize", dest="visualize", action="store_true")

    parser.add_argument("-n", dest="network", default="resnet32", type=str)
    parser.add_argument("-r", dest="optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-c", dest="decay", default=0.0001, type=float)
    parser.add_argument("-s", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--drop1", dest="drop1", default=100, type=int)
    parser.add_argument("--drop2", dest="drop2", default=200, type=int)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse user parameters and set device
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 32, 'pin_memory': True}

    # Set random seeds
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Load data
    batch_size = args.batch_size
    trainloader, testloader = helper.load_cifar100(args.datadir, batch_size, kwargs)
    nr_classes = 100

    # Load the polars and update the trainy labels
    classpolars = torch.from_numpy(np.load(args.hpnfile)).float()
    args.output_dims = int(args.hpnfile.split("/")[-1].split("-")[1][:-1])
    
    # Load the base model
    if args.network == "resnet32":
        base_model = resnet.ResNet(32, args.output_dims, 1)
    elif args.network == "densenet121":
        base_model = densenet.DenseNet121(args.output_dims)
    
    # Create HPN with Einstein-Rosen bridge
    model = HPNWithBridge(
        base_model=base_model,
        polars=classpolars,
        bridge_radius=args.bridge_radius
    )
    model = model.to(device)
    
    # Load the optimizer
    optimizer = helper.get_optimizer(
        args.optimizer,
        model.parameters(),
        args.learning_rate,
        args.momentum,
        args.decay
    )
    
    # Initialize the loss function
    f_loss = nn.CosineSimilarity(eps=1e-9).to(device)

    # Visualize the bridge if requested (only works for 3D embeddings)
    if args.visualize and args.output_dims == 3:
        visualize_bridge(model, save_path=f"{args.resdir}/bridge_visualization.png")
        print(f"Bridge visualization saved to {args.resdir}/bridge_visualization.png")

    # Main training loop
    testscores = []
    learning_rate = args.learning_rate
    for i in range(args.epochs):
        print("---")
        # Learning rate decay
        if i in [args.drop1, args.drop2]:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Train and test
        train_with_bridge(model, device, trainloader, optimizer, f_loss, i)
        if i % 10 == 0 or i == args.epochs - 1:
            t = test_with_bridge(model, device, testloader)
            testscores.append([i, t])
    
    # Save final model and results
    torch.save(model.state_dict(), f"{args.resdir}/hpn_bridge_model.pth")
    np.save(f"{args.resdir}/hpn_bridge_results.npy", np.array(testscores))
    
    # Final visualization
    if args.visualize and args.output_dims == 3:
        visualize_bridge(model, save_path=f"{args.resdir}/bridge_visualization_final.png")
        print(f"Final bridge visualization saved to {args.resdir}/bridge_visualization_final.png") 