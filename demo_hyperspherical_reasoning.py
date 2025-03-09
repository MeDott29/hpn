"""
Demo of Hyperspherical Graph Reasoning

This script demonstrates how to use the HypersphericalGraphReasoner with a real LLM.
It shows how to initialize the reasoner, perform iterative reasoning, and analyze the results.
"""

import os
import sys
import torch
import argparse
from hyperspherical_graph_reasoner import HypersphericalGraphReasoner

# Import your preferred LLM library here
# For example, if using OpenAI:
# import openai
# from openai import OpenAI

class OpenAILLM:
    """
    A wrapper for the OpenAI API to use with the HypersphericalGraphReasoner.
    
    This is just an example. You can replace this with any LLM API you prefer.
    """
    def __init__(self, model_name="gpt-4"):
        """
        Initialize the OpenAI LLM.
        
        Args:
            model_name: The name of the OpenAI model to use
        """
        # Uncomment and modify this code to use the actual OpenAI API
        # self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        
    def generate(self, prompt):
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The generated response
        """
        # Uncomment and modify this code to use the actual OpenAI API
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7,
        #     max_tokens=1500
        # )
        # return response.choices[0].message.content
        
        # For demonstration purposes, return a dummy response
        if "<|thinking|>" in prompt:
            return """<|thinking|>
Let me think about this question...

<graph>
Artificial Intelligence -- IS-A --> Field of Study
Artificial Intelligence -- RELATES-TO --> Machine Learning
Machine Learning -- USES --> Neural Networks
Neural Networks -- ENABLE --> Deep Learning
Deep Learning -- APPLIES-TO --> Computer Vision
Computer Vision -- ENABLES --> Object Recognition
Object Recognition -- USED-IN --> Autonomous Vehicles
Autonomous Vehicles -- REQUIRE --> Sensor Fusion
Sensor Fusion -- COMBINES --> LiDAR
Sensor Fusion -- COMBINES --> Radar
Sensor Fusion -- COMBINES --> Camera Data
</graph>

Based on the above knowledge graph, I can see that Artificial Intelligence is a field of study that relates to Machine Learning. Machine Learning uses Neural Networks, which enable Deep Learning. Deep Learning applies to Computer Vision, which enables Object Recognition. Object Recognition is used in Autonomous Vehicles, which require Sensor Fusion. Sensor Fusion combines data from LiDAR, Radar, and Cameras.
<|/thinking|>"""
        else:
            return "How do neural networks contribute to advancements in autonomous vehicle technology?"


class AnthropicLLM:
    """
    A wrapper for the Anthropic API to use with the HypersphericalGraphReasoner.
    """
    def __init__(self, model_name="claude-3-opus-20240229"):
        """
        Initialize the Anthropic LLM.
        
        Args:
            model_name: The name of the Anthropic model to use
        """
        # Uncomment and modify this code to use the actual Anthropic API
        # import anthropic
        # self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model_name = model_name
        
    def generate(self, prompt):
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The generated response
        """
        # Uncomment and modify this code to use the actual Anthropic API
        # response = self.client.messages.create(
        #     model=self.model_name,
        #     max_tokens=1500,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.content[0].text
        
        # For demonstration purposes, return a dummy response
        if "<|thinking|>" in prompt:
            return """<|thinking|>
Let me analyze this question carefully...

<graph>
Hyperspherical Embeddings -- IS-A --> Representation Technique
Hyperspherical Embeddings -- CONSTRAINS --> Vector Norm
Vector Norm -- EQUALS --> Constant Value
Hyperspherical Embeddings -- ENABLES --> Cosine Similarity
Cosine Similarity -- MEASURES --> Directional Similarity
Directional Similarity -- IGNORES --> Magnitude Differences
Hyperspherical Embeddings -- USED-IN --> Face Recognition
Hyperspherical Embeddings -- USED-IN --> Document Classification
Hyperspherical Embeddings -- RELATES-TO --> Manifold Learning
Manifold Learning -- STUDIES --> Data Geometry
</graph>

Based on the knowledge graph above, I can see that hyperspherical embeddings are a representation technique that constrains vector norms to a constant value. This enables the use of cosine similarity, which measures directional similarity while ignoring magnitude differences. Hyperspherical embeddings are used in applications like face recognition and document classification, and they relate to manifold learning, which studies the geometry of data.
<|/thinking|>"""
        else:
            return "What are the advantages of hyperspherical embeddings in deep learning architectures?"


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description="Demo of Hyperspherical Graph Reasoning")
    parser.add_argument("--model", type=str, default="openai", choices=["openai", "anthropic"],
                        help="The LLM provider to use")
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help="The dimensionality of node embeddings")
    parser.add_argument("--bridge_radius", type=float, default=0.3,
                        help="The radius of the Einstein-Rosen bridge")
    parser.add_argument("--iterations", type=int, default=5,
                        help="The number of reasoning iterations")
    parser.add_argument("--question", type=str, default="What are the applications of artificial intelligence?",
                        help="The initial question to start reasoning from")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The directory to save output files")
    return parser.parse_args()


def main():
    """
    Main function to run the demo.
    """
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the LLM
    if args.model == "openai":
        llm = OpenAILLM()
    elif args.model == "anthropic":
        llm = AnthropicLLM()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Initialize the reasoner
    reasoner = HypersphericalGraphReasoner(
        llm_model=llm,
        embedding_dim=args.embedding_dim,
        bridge_radius=args.bridge_radius
    )
    
    # Perform iterative reasoning
    print(f"Starting iterative reasoning with {args.iterations} iterations...")
    print(f"Initial question: {args.question}")
    reasoner.iterative_reasoning(
        initial_question=args.question,
        max_iterations=args.iterations
    )
    
    # Analyze the results
    print("\nAnalyzing the knowledge graph...")
    bridge_nodes = reasoner.identify_bridge_nodes()
    hubs = reasoner.identify_hubs()
    
    print(f"Number of nodes: {len(reasoner.graph)}")
    print(f"Number of bridge nodes: {len(bridge_nodes)}")
    print(f"Top 5 bridge nodes: {sorted(bridge_nodes.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Top 5 hubs: {sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    # Visualize the graph
    print("\nVisualizing the knowledge graph...")
    reasoner.visualize_graph(os.path.join(args.output_dir, "knowledge_graph.png"))
    
    # Save the graph
    print("\nSaving the knowledge graph...")
    reasoner.save_graph(os.path.join(args.output_dir, "knowledge_graph.json"))
    
    print(f"\nDemo completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 