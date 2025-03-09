# Hyperspherical Graph Reasoner

This project implements an agentic deep graph reasoning system with hyperspherical embeddings. It combines the strengths of both approaches to create a more robust knowledge representation system that enforces geometric constraints on node embeddings.

## Overview

The Hyperspherical Graph Reasoner is an autonomous graph expansion framework that iteratively structures and refines knowledge in situ. Unlike conventional knowledge graph construction methods relying on static extraction or single-pass learning, our approach couples a reasoning-native large language model with a continually updated graph representation.

At each step, the system:
1. Actively generates new concepts and relationships
2. Merges them into a global graph
3. Optimizes node embeddings on a hypersphere
4. Formulates subsequent prompts based on the evolving structure

Through this feedback-driven loop, the model organizes information into a scale-free network characterized by hub formation, stable modularity, and bridging nodes that link disparate knowledge clusters.

## Key Features

- **Hyperspherical Embeddings**: All node representations are constrained to lie on the unit hypersphere, ensuring that similarity is based on direction rather than magnitude.
- **Einstein-Rosen Bridge**: A novel mechanism that creates wormhole-like connections between antipodal points on the hypersphere, allowing for more efficient information transfer.
- **Iterative Reasoning**: The system continuously expands and refines the knowledge graph through recursive reasoning.
- **Bridge Node Detection**: Automatically identifies nodes that connect different knowledge domains.
- **Hub Identification**: Detects central concepts that have many connections.
- **Visualization**: Provides tools to visualize the knowledge graph and analyze its structure.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hyperspherical-graph-reasoner.git
cd hyperspherical-graph-reasoner

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from hyperspherical_graph_reasoner import HypersphericalGraphReasoner

# Initialize the reasoner with your LLM
reasoner = HypersphericalGraphReasoner(
    llm_model=your_llm_model,
    embedding_dim=64,
    bridge_radius=0.3
)

# Perform iterative reasoning
reasoner.iterative_reasoning(
    initial_question="What are the applications of artificial intelligence?",
    max_iterations=10
)

# Analyze the results
bridge_nodes = reasoner.identify_bridge_nodes()
hubs = reasoner.identify_hubs()

# Visualize the graph
reasoner.visualize_graph("knowledge_graph.png")

# Save the graph
reasoner.save_graph("knowledge_graph.json")
```

### Demo Script

We provide a demo script that shows how to use the Hyperspherical Graph Reasoner with different LLM providers:

```bash
python demo_hyperspherical_reasoning.py --model openai --iterations 5 --question "What are the applications of artificial intelligence?"
```

Command-line arguments:
- `--model`: The LLM provider to use (openai or anthropic)
- `--embedding_dim`: The dimensionality of node embeddings
- `--bridge_radius`: The radius of the Einstein-Rosen bridge
- `--iterations`: The number of reasoning iterations
- `--question`: The initial question to start reasoning from
- `--output_dir`: The directory to save output files

## How It Works

### 1. Knowledge Representation

Nodes in the knowledge graph are represented as points on a hypersphere, ensuring that all embeddings have the same norm. This focuses similarity on direction rather than magnitude, which has been shown to be beneficial for many knowledge representation tasks.

### 2. Einstein-Rosen Bridge

The Einstein-Rosen bridge creates a wormhole-like connection between antipodal points on the hypersphere. This allows for more efficient information transfer and can help the system discover connections between seemingly unrelated concepts.

### 3. Iterative Reasoning

The system performs iterative reasoning by:
1. Generating reasoning text using an LLM
2. Extracting a local graph from the reasoning
3. Merging the local graph with the global graph
4. Optimizing node embeddings on the hypersphere
5. Generating a follow-up question based on recent additions

### 4. Embedding Optimization

Node embeddings are optimized to respect two constraints:
1. Related nodes should have high similarity (cosine similarity)
2. Unrelated nodes should have low similarity

This is achieved through gradient descent with a loss function that maximizes similarity between related nodes and minimizes similarity between unrelated nodes.

## Applications

The Hyperspherical Graph Reasoner can be used for various applications, including:

- **Knowledge Discovery**: Discover new connections between concepts in a domain.
- **Question Answering**: Build a knowledge graph to answer complex questions.
- **Research Assistance**: Help researchers explore and organize knowledge in their field.
- **Education**: Create structured knowledge representations for educational purposes.

## Requirements

- Python 3.8+
- PyTorch
- NetworkX
- Matplotlib
- An LLM provider (OpenAI, Anthropic, etc.)

## Citation

If you use this code in your research, please cite:

```
@article{hyperspherical_graph_reasoner,
  title={Agentic Deep Graph Reasoning with Hyperspherical Embeddings},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 