#!/usr/bin/env python3
"""
Project Aurelia - Simplified Architecture Diagram Generator
Creates two focused diagrams: High-Level Architecture and Data Architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

class SimpleArchitectureDiagram:
    """Generate simplified architecture diagrams for Project Aurelia"""
    
    def __init__(self):
        self.colors = {
            'frontend': '#E3F2FD',      # Light blue
            'backend': '#F3E5F5',       # Light purple
            'services': '#E8F5E8',       # Light green
            'data': '#FFF3E0',          # Light orange
            'external': '#FFEBEE',       # Light red
            'storage': '#F1F8E9',       # Light lime
            'border': '#1976D2',        # Blue
            'text': '#212121'           # Dark gray
        }
        
    def create_high_level_architecture(self, save_path: str = "high_level_architecture.png"):
        """Create high-level architecture diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(6, 9.5, 'Project Aurelia - High-Level Architecture', 
                fontsize=18, fontweight='bold', ha='center', va='center')
        
        # Frontend Layer
        self._draw_layer(ax, 1, 7.5, 2, 1.5, 'Frontend Layer', self.colors['frontend'])
        self._draw_component(ax, 1.2, 7.8, 0.8, 0.8, 'Streamlit\nChat UI', self.colors['frontend'])
        self._draw_component(ax, 2.2, 7.8, 0.8, 0.8, 'Configuration\nPanel', self.colors['frontend'])
        
        # Backend Layer
        self._draw_layer(ax, 1, 5.5, 2, 1.5, 'Backend Layer', self.colors['backend'])
        self._draw_component(ax, 1.2, 5.8, 0.8, 0.8, 'FastAPI\nServer', self.colors['backend'])
        self._draw_component(ax, 2.2, 5.8, 0.8, 0.8, 'API\nRouter', self.colors['backend'])
        
        # Query Processing
        self._draw_layer(ax, 1, 3.5, 2, 1.5, 'Query Processing', self.colors['services'])
        self._draw_component(ax, 1.2, 3.8, 0.8, 0.8, 'Query\nRewriter', self.colors['services'])
        self._draw_component(ax, 2.2, 3.8, 0.8, 0.8, 'Instructor\nClassifier', self.colors['services'])
        
        # Query Classification Types
        self._draw_layer(ax, 4, 3.5, 2, 1.5, 'Query Types', self.colors['services'])
        self._draw_component(ax, 4.2, 4.2, 0.6, 0.5, 'External\n(Personal)', self.colors['external'])
        self._draw_component(ax, 4.2, 3.5, 0.6, 0.5, 'Documental\n(Finance)', self.colors['data'])
        self._draw_component(ax, 5.2, 3.8, 0.6, 0.5, 'Context\n(Conversation)', self.colors['frontend'])
        
        # RAG Pipeline
        self._draw_layer(ax, 7, 5.5, 2, 1.5, 'RAG Pipeline', self.colors['services'])
        self._draw_component(ax, 7.2, 5.8, 0.8, 0.8, 'Vector\nSearch', self.colors['services'])
        self._draw_component(ax, 8.2, 5.8, 0.8, 0.8, 'Hybrid\nSearch', self.colors['services'])
        
        # Generation Services
        self._draw_layer(ax, 7, 3.5, 2, 1.5, 'Generation Services', self.colors['services'])
        self._draw_component(ax, 7.2, 3.8, 0.8, 0.8, 'OpenAI\nGeneration', self.colors['services'])
        self._draw_component(ax, 8.2, 3.8, 0.8, 0.8, 'ChatGPT\nService', self.colors['services'])
        
        # Storage Layer
        self._draw_layer(ax, 10, 5.5, 1.5, 1.5, 'Storage Layer', self.colors['storage'])
        self._draw_component(ax, 10.2, 5.8, 0.8, 0.8, 'Pinecone\nVector DB', self.colors['storage'])
        
        self._draw_layer(ax, 10, 3.5, 1.5, 1.5, 'Cache & Context', self.colors['storage'])
        self._draw_component(ax, 10.2, 3.8, 0.8, 0.8, 'Redis\nCache', self.colors['storage'])
        
        # External Services
        self._draw_layer(ax, 1, 1, 2, 1.5, 'External Services', self.colors['external'])
        self._draw_component(ax, 1.2, 1.3, 0.8, 0.8, 'OpenAI\nAPI', self.colors['external'])
        self._draw_component(ax, 2.2, 1.3, 0.8, 0.8, 'Wikipedia\nAPI', self.colors['external'])
        
        # Airflow & GCS
        self._draw_layer(ax, 7, 1, 2, 1.5, 'Orchestration', self.colors['external'])
        self._draw_component(ax, 7.2, 1.3, 0.8, 0.8, 'Airflow\nDAG', self.colors['external'])
        self._draw_component(ax, 8.2, 1.3, 0.8, 0.8, 'Google\nCloud Storage', self.colors['external'])
        
        # Add connections
        self._draw_high_level_connections(ax)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"High-level architecture diagram saved as: {save_path}")
        return fig
    
    def create_data_architecture(self, save_path: str = "data_architecture.png"):
        """Create data architecture diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Title
        ax.text(8, 7.5, 'Project Aurelia - Data Architecture & Pipeline', 
                fontsize=18, fontweight='bold', ha='center', va='center')
        
        # Data Ingestion Pipeline (Top)
        ax.text(8, 6.5, 'Data Ingestion Pipeline', fontsize=14, fontweight='bold', 
                ha='center', va='center')
        
        ingestion_steps = [
            (1, 5.5, 'PDF\nDocuments', self.colors['data']),
            (3, 5.5, 'Docling\nParser', self.colors['data']),
            (5, 5.5, 'PyMuPDF\nParser', self.colors['data']),
            (7, 5.5, 'Hybrid\nChunking', self.colors['data']),
            (9, 5.5, 'Embedding\nGeneration', self.colors['data']),
            (11, 5.5, 'Vector\nStorage', self.colors['storage']),
            (13, 5.5, 'Airflow\nDAG', self.colors['external']),
            (15, 5.5, 'Google\nCloud Storage', self.colors['external']),
        ]
        
        # Draw ingestion steps
        for x, y, text, color in ingestion_steps:
            self._draw_component(ax, x-0.4, y-0.3, 0.8, 0.6, text, color)
        
        # Draw arrows between ingestion steps
        for i in range(len(ingestion_steps)-1):
            start_x = ingestion_steps[i][0] + 0.4
            end_x = ingestion_steps[i+1][0] - 0.4
            y = ingestion_steps[i][1]
            arrow = ConnectionPatch((start_x, y), (end_x, y), "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5,
                                  mutation_scale=20, fc='black', ec='black')
            ax.add_patch(arrow)
        
        # Query Processing Flow (Middle)
        ax.text(8, 4, 'Query Processing & Classification Flow', fontsize=14, fontweight='bold', 
                ha='center', va='center')
        
        # Query flow steps
        query_steps = [
            (2, 3, 'User Query\n(Streamlit)', self.colors['frontend']),
            (4, 3, 'Query\nRewriter', self.colors['services']),
            (6, 3, 'Instructor\nClassifier', self.colors['services']),
            (8, 3, 'Query\nRouter', self.colors['services']),
        ]
        
        # Draw query steps
        for x, y, text, color in query_steps:
            self._draw_component(ax, x-0.4, y-0.3, 0.8, 0.6, text, color)
        
        # Draw arrows between query steps
        for i in range(len(query_steps)-1):
            start_x = query_steps[i][0] + 0.4
            end_x = query_steps[i+1][0] - 0.4
            y = query_steps[i][1]
            arrow = ConnectionPatch((start_x, y), (end_x, y), "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5,
                                  mutation_scale=20, fc='black', ec='black')
            ax.add_patch(arrow)
        
        # Query Classification Branches
        ax.text(10, 3, 'External\n(Personal)', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=self.colors['external']))
        ax.text(12, 3, 'Documental\n(Finance)', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=self.colors['data']))
        ax.text(14, 3, 'Context\n(Conversation)', fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=self.colors['frontend']))
        
        # RAG Pipeline (Bottom Left)
        ax.text(4, 1.5, 'RAG Pipeline (Documental Queries)', fontsize=12, fontweight='bold', 
                ha='center', va='center')
        
        rag_steps = [
            (1, 0.5, 'Vector\nSearch', self.colors['services']),
            (3, 0.5, 'Hybrid\nSearch', self.colors['services']),
            (5, 0.5, 'Neural\nReranking', self.colors['services']),
            (7, 0.5, 'OpenAI\nGeneration', self.colors['services']),
        ]
        
        # Draw RAG steps
        for x, y, text, color in rag_steps:
            self._draw_component(ax, x-0.4, y-0.3, 0.8, 0.6, text, color)
        
        # Draw arrows between RAG steps
        for i in range(len(rag_steps)-1):
            start_x = rag_steps[i][0] + 0.4
            end_x = rag_steps[i+1][0] - 0.4
            y = rag_steps[i][1]
            arrow = ConnectionPatch((start_x, y), (end_x, y), "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5,
                                  mutation_scale=20, fc='black', ec='black')
            ax.add_patch(arrow)
        
        # Bypass Flow (Bottom Right)
        ax.text(12, 1.5, 'Bypass Flow (External Queries)', fontsize=12, fontweight='bold', 
                ha='center', va='center')
        
        bypass_steps = [
            (10, 0.5, 'ChatGPT\nService', self.colors['services']),
            (12, 0.5, 'Direct\nResponse', self.colors['services']),
        ]
        
        # Draw bypass steps
        for x, y, text, color in bypass_steps:
            self._draw_component(ax, x-0.4, y-0.3, 0.8, 0.6, text, color)
        
        # Draw arrow between bypass steps
        arrow = ConnectionPatch((10.4, 0.5), (11.6, 0.5), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc='red', ec='red')
        ax.add_patch(arrow)
        
        # Add key connections
        # Vector Storage to RAG Pipeline
        arrow = ConnectionPatch((11, 5.2), (1, 0.8), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc='blue', ec='blue', alpha=0.7)
        ax.add_patch(arrow)
        
        # Query Router to RAG Pipeline
        arrow = ConnectionPatch((8.4, 2.7), (1, 0.8), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc='green', ec='green', alpha=0.7)
        ax.add_patch(arrow)
        
        # External queries bypass
        arrow = ConnectionPatch((10, 2.7), (10, 0.8), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=15, fc='red', ec='red', alpha=0.7)
        ax.add_patch(arrow)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data architecture diagram saved as: {save_path}")
        return fig
    
    def _draw_layer(self, ax, x, y, width, height, title, color):
        """Draw a layer container"""
        rect = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor=self.colors['border'],
                             linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height - 0.2, title, 
                fontsize=12, fontweight='bold', ha='center', va='center')
    
    def _draw_component(self, ax, x, y, width, height, text, color):
        """Draw a component box"""
        rect = FancyBboxPatch((x, y), width, height,
                             boxstyle="round,pad=0.05",
                             facecolor=color,
                             edgecolor=self.colors['border'],
                             linewidth=1)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, 
                fontsize=9, ha='center', va='center', wrap=True)
    
    def _draw_high_level_connections(self, ax):
        """Draw connections for high-level architecture"""
        connections = [
            # Frontend to Backend
            ((1.6, 7.5), (1.6, 5.5)),
            ((2.6, 7.5), (2.6, 5.5)),
            
            # Backend to Query Processing
            ((1.6, 5.5), (1.6, 3.5)),
            ((2.6, 5.5), (2.6, 3.5)),
            
            # Query Processing to Classification
            ((3.2, 3.8), (4, 3.8)),
            
            # Classification to RAG Pipeline (Documental)
            ((4.2, 3.5), (7, 5.8)),
            
            # Classification to Generation (External)
            ((4.2, 4.2), (8.2, 3.8)),
            
            # RAG Pipeline to Storage
            ((9, 5.8), (10, 5.8)),
            
            # Generation to External APIs
            ((7.2, 3.5), (1.2, 1.3)),
            ((8.2, 3.5), (2.2, 1.3)),
            
            # Orchestration to Storage
            ((8.2, 1.3), (15, 5.2)),
        ]
        
        for start, end in connections:
            arrow = ConnectionPatch(start, end, "data", "data",
                                  arrowstyle="->", shrinkA=5, shrinkB=5,
                                  mutation_scale=20, fc=self.colors['border'],
                                  ec=self.colors['border'], linewidth=1.5)
            ax.add_patch(arrow)
    
    def generate_diagrams(self):
        """Generate both diagrams"""
        print("🏗️  Generating Project Aurelia Architecture Diagrams...")
        print("=" * 60)
        
        # Create output directory
        os.makedirs("architecture_diagrams", exist_ok=True)
        
        # Generate high-level architecture
        print("📊 Creating high-level architecture diagram...")
        self.create_high_level_architecture("architecture_diagrams/high_level_architecture.png")
        
        # Generate data architecture
        print("🔄 Creating data architecture diagram...")
        self.create_data_architecture("architecture_diagrams/data_architecture.png")
        
        print("=" * 60)
        print("✅ Both diagrams generated successfully!")
        print("📁 Check the 'architecture_diagrams' folder for generated files.")
        
        # Print summary
        print("\n📋 Generated Files:")
        print("  • high_level_architecture.png - System overview")
        print("  • data_architecture.png - Data pipeline flow")

def main():
    """Main function to generate architecture diagrams"""
    print("🚀 Project Aurelia - Simplified Architecture Diagram Generator")
    print("=" * 60)
    
    # Check dependencies
    try:
        import matplotlib
        print("✅ matplotlib available")
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    # Generate diagrams
    diagram_generator = SimpleArchitectureDiagram()
    diagram_generator.generate_diagrams()
    
    print("\n🎉 Architecture diagram generation complete!")

if __name__ == "__main__":
    main()
