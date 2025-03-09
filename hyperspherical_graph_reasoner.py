"""
Hyperspherical Graph Reasoner

This module implements an agentic deep graph reasoning system with hyperspherical embeddings.
It combines the strengths of both approaches to create a more robust knowledge representation
system that enforces geometric constraints on node embeddings.

The system iteratively expands a knowledge graph through reasoning, while maintaining
hyperspherical constraints on node embeddings to improve relationship quality.
"""

import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import re
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Union, Any

class HypersphericalKnowledgeNode:
    """
    A node in the knowledge graph with a hyperspherical embedding.
    
    The embedding is constrained to lie on the unit hypersphere, ensuring
    that all node representations have the same norm and focusing similarity
    on direction rather than magnitude.
    """
    def __init__(self, concept_name: str, embedding_dim: int = 128):
        """
        Initialize a node with a random unit vector embedding.
        
        Args:
            concept_name: The name of the concept represented by this node
            embedding_dim: The dimensionality of the embedding vector
        """
        self.name = concept_name
        self.embedding = torch.randn(embedding_dim)
        self.embedding = F.normalize(self.embedding, p=2, dim=0)
        self.connections = {}  # Maps connected nodes to relation types
        self.attributes = {}   # Additional attributes for the node
        
    def update_embedding(self, new_embedding: torch.Tensor):
        """
        Update the node's embedding while ensuring it stays on the hypersphere.
        
        Args:
            new_embedding: The new embedding vector (will be normalized)
        """
        self.embedding = F.normalize(new_embedding, p=2, dim=0)
        
    def add_connection(self, target_node_name: str, relation_type: str):
        """
        Add a connection to another node with a specified relation type.
        
        Args:
            target_node_name: The name of the target node
            relation_type: The type of relation (e.g., "IS-A", "RELATES-TO")
        """
        self.connections[target_node_name] = relation_type
        
    def add_attribute(self, key: str, value: Any):
        """
        Add an attribute to the node.
        
        Args:
            key: The attribute name
            value: The attribute value
        """
        self.attributes[key] = value
        
    def __repr__(self):
        return f"Node({self.name}, {len(self.connections)} connections)"


class EinsteinRosenBridge(nn.Module):
    """
    Implementation of an Einstein-Rosen bridge for hyperspherical spaces.
    
    The bridge creates a wormhole-like connection between antipodal points
    on the hypersphere, allowing for more efficient information transfer.
    """
    def __init__(self, embedding_dim: int, bridge_radius: float = 0.3):
        super(EinsteinRosenBridge, self).__init__()
        self.embedding_dim = embedding_dim
        self.bridge_radius = bridge_radius
        
        # Bridge parameters
        self.pole_vectors = nn.Parameter(torch.randn(2, embedding_dim))
        
        # Normalize pole vectors to unit length
        with torch.no_grad():
            self.pole_vectors.div_(torch.norm(self.pole_vectors, dim=1, keepdim=True))
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Einstein-Rosen bridge.
        
        Args:
            embeddings: Input tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Transformed tensor of shape (batch_size, embedding_dim)
        """
        # Ensure input is normalized
        x_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate distance to poles
        north_dist = 1 - F.cosine_similarity(x_norm, self.pole_vectors[0].unsqueeze(0), dim=1)
        south_dist = 1 - F.cosine_similarity(x_norm, self.pole_vectors[1].unsqueeze(0), dim=1)
        
        # Determine if points are within bridge radius of either pole
        north_mask = (north_dist < self.bridge_radius).float().unsqueeze(1)
        south_mask = (south_dist < self.bridge_radius).float().unsqueeze(1)
        
        # Apply bridge transformation for points near poles
        # For points near north pole, connect to area near south pole and vice versa
        bridge_effect = north_mask * (self.pole_vectors[1] - x_norm * north_dist.unsqueeze(1)) + \
                        south_mask * (self.pole_vectors[0] - x_norm * south_dist.unsqueeze(1))
        
        # Combine original points with bridge effect
        bridge_mask = (north_mask + south_mask).clamp(0, 1)
        output = (1 - bridge_mask) * x_norm + bridge_mask * bridge_effect
        
        # Ensure output is normalized
        output = F.normalize(output, p=2, dim=1)
        
        return output


class HypersphericalGraphReasoner:
    """
    A graph reasoner that uses hyperspherical embeddings to represent knowledge.
    
    This class implements an agentic, autonomous graph expansion framework that
    iteratively structures and refines knowledge in situ, while maintaining
    hyperspherical constraints on node embeddings.
    """
    def __init__(self, llm_model, embedding_dim: int = 128, bridge_radius: float = 0.3):
        """
        Initialize the graph reasoner.
        
        Args:
            llm_model: The language model to use for reasoning
            embedding_dim: The dimensionality of node embeddings
            bridge_radius: The radius of the Einstein-Rosen bridge
        """
        self.llm = llm_model
        self.embedding_dim = embedding_dim
        self.graph = {}  # Map from concept names to HypersphericalKnowledgeNodes
        self.bridge = EinsteinRosenBridge(embedding_dim, bridge_radius)
        self.nx_graph = nx.DiGraph()  # NetworkX graph for analysis
        
    def extract_local_graph(self, reasoning_text: str) -> Dict[str, HypersphericalKnowledgeNode]:
        """
        Extract a local graph from the reasoning text.
        
        Args:
            reasoning_text: The reasoning text from the LLM
            
        Returns:
            A dictionary mapping concept names to nodes
        """
        local_graph = {}
        
        # Extract the graph section from the reasoning text
        graph_match = re.search(r'<graph>(.*?)</graph>', reasoning_text, re.DOTALL)
        if not graph_match:
            return local_graph
            
        graph_text = graph_match.group(1).strip()
        
        # Parse the graph text to extract nodes and relations
        # Format: node1 -- RELATION --> node2
        relation_pattern = r'(.*?)\s+--\s+(.*?)\s+-->\s+(.*?)$'
        
        for line in graph_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = re.match(relation_pattern, line)
            if match:
                source, relation, target = match.groups()
                source = source.strip()
                relation = relation.strip()
                target = target.strip()
                
                # Create nodes if they don't exist
                for node_name in [source, target]:
                    if node_name not in local_graph:
                        if node_name in self.graph:
                            # Use existing node
                            local_graph[node_name] = self.graph[node_name]
                        else:
                            # Create new node
                            local_graph[node_name] = HypersphericalKnowledgeNode(
                                node_name, self.embedding_dim
                            )
                
                # Add the relation
                local_graph[source].add_connection(target, relation)
        
        return local_graph
    
    def merge_graphs(self, local_graph: Dict[str, HypersphericalKnowledgeNode]):
        """
        Merge the local graph into the global graph.
        
        Args:
            local_graph: The local graph to merge
        """
        # Add new nodes and update existing ones
        for concept, node in local_graph.items():
            if concept not in self.graph:
                self.graph[concept] = node
            else:
                # Update connections for existing node
                for connected, relation in node.connections.items():
                    self.graph[concept].add_connection(connected, relation)
        
        # Update the NetworkX graph for analysis
        self.update_nx_graph()
    
    def update_nx_graph(self):
        """
        Update the NetworkX graph based on the current state of the graph.
        """
        self.nx_graph = nx.DiGraph()
        
        # Add nodes
        for concept, node in self.graph.items():
            self.nx_graph.add_node(concept, embedding=node.embedding.numpy())
            
        # Add edges
        for concept, node in self.graph.items():
            for target, relation in node.connections.items():
                if target in self.graph:  # Ensure target exists
                    self.nx_graph.add_edge(concept, target, relation=relation)
    
    def optimize_embeddings(self, num_iterations: int = 10, learning_rate: float = 0.01):
        """
        Optimize node embeddings to respect both:
        1. Related nodes should have high similarity
        2. Unrelated nodes should have low similarity
        
        Args:
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
        """
        if not self.graph:
            return
            
        # Collect all nodes and their related/unrelated peers
        node_relations = []
        for node_name, node in self.graph.items():
            related = list(node.connections.keys())
            unrelated = [n for n in self.graph if n not in related and n != node_name]
            node_relations.append((node, related, unrelated))
        
        # Extract embeddings for optimization
        embeddings = {name: node.embedding.clone().requires_grad_(True) 
                     for name, node in self.graph.items()}
        
        # Optimize using gradient descent
        optimizer = optim.Adam(embeddings.values(), lr=learning_rate)
        
        for epoch in range(num_iterations):
            optimizer.zero_grad()
            
            loss = 0
            for node, related, unrelated in node_relations:
                # Maximize similarity with related nodes
                for r in related:
                    if r in embeddings:  # Ensure related node exists
                        sim = F.cosine_similarity(
                            embeddings[node.name].unsqueeze(0),
                            embeddings[r].unsqueeze(0),
                            dim=1
                        )
                        loss -= sim.mean()
                
                # Minimize similarity with unrelated nodes (sample a few)
                sampled_unrelated = random.sample(unrelated, min(5, len(unrelated))) if unrelated else []
                for u in sampled_unrelated:
                    if u in embeddings:  # Ensure unrelated node exists
                        sim = F.cosine_similarity(
                            embeddings[node.name].unsqueeze(0),
                            embeddings[u].unsqueeze(0),
                            dim=1
                        )
                        loss += torch.clamp(sim - torch.tensor(0.3), min=0).mean()
            
            loss.backward()
            optimizer.step()
            
            # Re-normalize embeddings to stay on hypersphere
            with torch.no_grad():
                for name, emb in embeddings.items():
                    embeddings[name] = F.normalize(emb, p=2, dim=0)
        
        # Apply Einstein-Rosen bridge transformation
        with torch.no_grad():
            all_embeddings = torch.stack(list(embeddings.values()))
            transformed_embeddings = self.bridge(all_embeddings)
            
            # Update node embeddings with transformed values
            for i, (name, _) in enumerate(embeddings.items()):
                self.graph[name].update_embedding(transformed_embeddings[i])
    
    def identify_bridge_nodes(self, community_detection_algo=None):
        """
        Identify bridge nodes in the graph.
        
        Args:
            community_detection_algo: The community detection algorithm to use
                                     (defaults to Louvain method)
                                     
        Returns:
            A dictionary mapping node names to bridge scores
        """
        if community_detection_algo is None:
            # Use Louvain method by default
            try:
                import community as community_louvain
                community_detection_algo = community_louvain.best_partition
            except ImportError:
                # Fallback to NetworkX's community detection
                from networkx.algorithms import community
                community_detection_algo = lambda g: {node: i for i, comm in enumerate(
                    community.greedy_modularity_communities(g.to_undirected())
                ) for node in comm}
        
        # Get communities
        communities = community_detection_algo(self.nx_graph.to_undirected())
        
        # Identify bridge nodes
        bridge_scores = {}
        for node_name in self.nx_graph.nodes():
            # Get neighbors
            neighbors = list(self.nx_graph.successors(node_name)) + list(self.nx_graph.predecessors(node_name))
            
            # Count communities
            neighbor_communities = {communities[neighbor] for neighbor in neighbors if neighbor in communities}
            
            if len(neighbor_communities) > 1:
                bridge_scores[node_name] = len(neighbor_communities)
        
        return bridge_scores
    
    def identify_hubs(self):
        """
        Identify hub nodes in the graph.
        
        Returns:
            A dictionary mapping node names to hub scores (degree centrality)
        """
        return {node: len(list(self.nx_graph.successors(node)) + list(self.nx_graph.predecessors(node)))
                for node in self.nx_graph.nodes()}
    
    def generate_next_question(self, recent_concepts: List[str]) -> str:
        """
        Generate a follow-up question based on recent concepts.
        
        Args:
            recent_concepts: List of recently added concepts
            
        Returns:
            A follow-up question
        """
        prompt = f"""Consider these concepts: {', '.join(recent_concepts)}. 
Formulate a creative follow-up question to ask about a totally new aspect.
Your question should include at least one of the original concepts.
Reply only with the new question."""
        
        response = self.llm.generate(prompt)
        return response.strip()
    
    def iterative_reasoning(self, initial_question: str, max_iterations: int = 100):
        """
        Perform iterative reasoning to expand the knowledge graph.
        
        Args:
            initial_question: The initial question to start reasoning from
            max_iterations: Maximum number of iterations
            
        Returns:
            The final knowledge graph
        """
        current_question = initial_question
        
        for i in range(max_iterations):
            print(f"Iteration {i+1}/{max_iterations}")
            print(f"Question: {current_question}")
            
            # Generate reasoning using LLM
            reasoning_prompt = f"<|thinking|>{current_question}\n\nPlease provide your reasoning and include a structured knowledge graph in <graph></graph> tags using the format: node1 -- RELATION --> node2\n<|/thinking|>"
            reasoning = self.llm.generate(reasoning_prompt)
            
            # Extract local graph from reasoning
            local_graph = self.extract_local_graph(reasoning)
            
            if not local_graph:
                print("No graph extracted, trying again with a different question.")
                current_question = f"Let's explore a different aspect of {current_question}"
                continue
            
            # Merge with global graph
            self.merge_graphs(local_graph)
            
            # Optimize embeddings
            self.optimize_embeddings()
            
            # Generate next question based on recent additions
            recent_concepts = list(local_graph.keys())
            current_question = self.generate_next_question(recent_concepts)
            
            # Analysis at regular intervals
            if i % 10 == 0:
                bridge_nodes = self.identify_bridge_nodes()
                hubs = self.identify_hubs()
                print(f"Number of nodes: {len(self.graph)}")
                print(f"Number of bridge nodes: {len(bridge_nodes)}")
                print(f"Top 5 hubs: {sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        return self.graph
    
    def visualize_graph(self, output_file: str = "knowledge_graph.png"):
        """
        Visualize the knowledge graph.
        
        Args:
            output_file: Path to save the visualization
        """
        plt.figure(figsize=(12, 12))
        
        # Use spring layout for node positions
        pos = nx.spring_layout(self.nx_graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.nx_graph, pos, node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.nx_graph, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(self.nx_graph, pos, font_size=10)
        
        # Draw edge labels
        edge_labels = {(u, v): d['relation'] for u, v, d in self.nx_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.nx_graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Graph visualization saved to {output_file}")
    
    def save_graph(self, output_file: str = "knowledge_graph.json"):
        """
        Save the knowledge graph to a JSON file.
        
        Args:
            output_file: Path to save the graph
        """
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Save nodes
        for name, node in self.graph.items():
            graph_data["nodes"].append({
                "id": name,
                "embedding": node.embedding.tolist(),
                "attributes": node.attributes
            })
        
        # Save edges
        for name, node in self.graph.items():
            for target, relation in node.connections.items():
                graph_data["edges"].append({
                    "source": name,
                    "target": target,
                    "relation": relation
                })
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Graph saved to {output_file}")
    
    def load_graph(self, input_file: str):
        """
        Load the knowledge graph from a JSON file.
        
        Args:
            input_file: Path to load the graph from
        """
        with open(input_file, 'r') as f:
            graph_data = json.load(f)
        
        # Clear existing graph
        self.graph = {}
        
        # Load nodes
        for node_data in graph_data["nodes"]:
            node = HypersphericalKnowledgeNode(node_data["id"], self.embedding_dim)
            node.embedding = torch.tensor(node_data["embedding"])
            node.attributes = node_data.get("attributes", {})
            self.graph[node_data["id"]] = node
        
        # Load edges
        for edge_data in graph_data["edges"]:
            source = edge_data["source"]
            target = edge_data["target"]
            relation = edge_data["relation"]
            
            if source in self.graph and target in self.graph:
                self.graph[source].add_connection(target, relation)
        
        # Update NetworkX graph
        self.update_nx_graph()
        
        print(f"Graph loaded from {input_file}")


# Example usage
if __name__ == "__main__":
    # This is a placeholder for the LLM model
    class DummyLLM:
        def generate(self, prompt):
            # In a real implementation, this would call the actual LLM
            return """<|thinking|>
Let me think about this question...

<graph>
Artificial Intelligence -- IS-A --> Field of Study
Artificial Intelligence -- RELATES-TO --> Machine Learning
Machine Learning -- USES --> Neural Networks
Neural Networks -- ENABLE --> Deep Learning
Deep Learning -- APPLIES-TO --> Computer Vision
Computer Vision -- ENABLES --> Object Recognition
</graph>

Based on the above knowledge graph, I can see that...
<|/thinking|>"""
    
    # Initialize the reasoner with a dummy LLM
    reasoner = HypersphericalGraphReasoner(DummyLLM(), embedding_dim=64)
    
    # Start iterative reasoning
    reasoner.iterative_reasoning("What are the applications of artificial intelligence?", max_iterations=3)
    
    # Visualize the graph
    reasoner.visualize_graph()
    
    # Save the graph
    reasoner.save_graph() 