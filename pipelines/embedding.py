#!/usr/bin/env python3
"""
Phase C - Embedding & Vector Storage Pipeline
Creates embeddings from chunks and stores them in Pinecone vector database
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import time


# Configuration
class Config:
    """Pipeline configuration"""
    # Paths
    CHUNKS_PATH = Path("data/chunking_output/hybrid_intelligent_chunks_for_embedding.jsonl")
    OUTPUT_DIR = Path("data/embedding_output")
    
    # Pinecone settings
    INDEX_NAME = "fintbx-embeddings"
    DIMENSION = 3072  # text-embedding-3-large
    METRIC = "cosine"
    CLOUD = "aws"
    REGION = "us-east-1"
    
    # Embedding settings
    EMBEDDING_MODEL = "text-embedding-3-large"
    BATCH_SIZE = 100  # Number of chunks to process at once
    PINECONE_BATCH_SIZE = 100  # Number of vectors to upsert at once
    
    # Rate limiting
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds


class EmbeddingPipeline:
    """Main embedding and vector storage pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.pinecone = None
        self.index = None
        self.stats = {
            'total_chunks': 0,
            'embedded_chunks': 0,
            'uploaded_vectors': 0,
            'failed_chunks': 0,
            'errors': []
        }
        
    def initialize_clients(self):
        """Initialize OpenAI and Pinecone clients"""
        print("🔧 Initializing clients...")
        
        # Load environment variables
        dotenv.load_dotenv()
        
        # Initialize OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=openai_key)
        print("✓ OpenAI client initialized")
        
        # Initialize Pinecone
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pinecone = Pinecone(api_key=pinecone_key)
        print("✓ Pinecone client initialized")
        
    def create_or_connect_index(self):
        """Create a new Pinecone index or connect to existing one"""
        print(f"\n📊 Setting up Pinecone index: {self.config.INDEX_NAME}")
        
        existing_indexes = [idx.name for idx in self.pinecone.list_indexes()]
        
        if self.config.INDEX_NAME in existing_indexes:
            print(f"✓ Index '{self.config.INDEX_NAME}' already exists, connecting...")
            self.index = self.pinecone.Index(self.config.INDEX_NAME)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            print(f"📊 Current index stats:")
            print(f"   - Total vectors: {stats.total_vector_count}")
            if stats.namespaces:
                print(f"   - Namespaces: {list(stats.namespaces.keys())}")
        else:
            print(f"Creating new index '{self.config.INDEX_NAME}'...")
            
            self.pinecone.create_index(
                name=self.config.INDEX_NAME,
                dimension=self.config.DIMENSION,
                metric=self.config.METRIC,
                spec=ServerlessSpec(
                    cloud=self.config.CLOUD,
                    region=self.config.REGION
                )
            )
            
            print(f"✅ Index created successfully!")
            print(f"   - Dimension: {self.config.DIMENSION}")
            print(f"   - Metric: {self.config.METRIC}")
            print(f"   - Region: {self.config.CLOUD}/{self.config.REGION}")
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            time.sleep(5)
            
            self.index = self.pinecone.Index(self.config.INDEX_NAME)
            
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSONL file"""
        print(f"\n📂 Loading chunks from: {self.config.CHUNKS_PATH}")
        
        if not self.config.CHUNKS_PATH.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.config.CHUNKS_PATH}")
        
        chunks = []
        with open(self.config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        
        self.stats['total_chunks'] = len(chunks)
        print(f"✓ Loaded {len(chunks)} chunks")
        
        return chunks
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a batch of texts with retry logic"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.config.EMBEDDING_MODEL
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt < self.config.MAX_RETRIES - 1:
                    print(f"⚠️  Retry {attempt + 1}/{self.config.MAX_RETRIES} after error: {e}")
                    time.sleep(self.config.RETRY_DELAY)
                else:
                    raise e
    
    def prepare_metadata(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for Pinecone storage"""
        metadata = chunk.get('metadata', {})
        
        # Convert complex types to strings for Pinecone
        return {
            "text": chunk['text'][:40000],  # Pinecone metadata limit
            "chunk_id": chunk['id'],
            "strategy": metadata.get('strategy', ''),
            "section": metadata.get('section', '')[:500],  # Limit length
            "pages": json.dumps(metadata.get('pages', [])),
            "is_special": metadata.get('is_special', False),
            "is_semantic": metadata.get('is_semantic', False),
            "char_count": metadata.get('char_count', 0),
            "word_count": metadata.get('word_count', 0),
            "source": metadata.get('source', ''),
            "content_types": json.dumps(metadata.get('content_types', [])),
        }
    
    def process_and_upload_chunks(self, chunks: List[Dict[str, Any]]):
        """Process chunks in batches: create embeddings and upload to Pinecone"""
        print(f"\n🚀 Processing {len(chunks)} chunks...")
        print(f"   Embedding model: {self.config.EMBEDDING_MODEL}")
        print(f"   Batch size: {self.config.BATCH_SIZE}")
        
        total_batches = (len(chunks) + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        for i in tqdm(range(0, len(chunks), self.config.BATCH_SIZE), 
                     desc="Processing batches", 
                     total=total_batches):
            batch = chunks[i:i + self.config.BATCH_SIZE]
            
            try:
                # Extract texts for embedding
                texts = [chunk['text'] for chunk in batch]
                
                # Create embeddings
                embeddings = self.create_embeddings_batch(texts)
                self.stats['embedded_chunks'] += len(embeddings)
                
                # Prepare vectors for Pinecone
                vectors_to_upsert = []
                for j, chunk in enumerate(batch):
                    vector = {
                        "id": str(chunk['id']),
                        "values": embeddings[j],
                        "metadata": self.prepare_metadata(chunk)
                    }
                    vectors_to_upsert.append(vector)
                
                # Upload to Pinecone in sub-batches
                for k in range(0, len(vectors_to_upsert), self.config.PINECONE_BATCH_SIZE):
                    sub_batch = vectors_to_upsert[k:k + self.config.PINECONE_BATCH_SIZE]
                    self.index.upsert(vectors=sub_batch)
                    self.stats['uploaded_vectors'] += len(sub_batch)
                
            except Exception as e:
                error_msg = f"Batch {i//self.config.BATCH_SIZE + 1}: {str(e)}"
                self.stats['errors'].append(error_msg)
                self.stats['failed_chunks'] += len(batch)
                print(f"\n❌ Error processing batch: {e}")
                continue
        
        print(f"\n✅ Processing complete!")
    
    def verify_upload(self):
        """Verify the upload was successful"""
        print("\n🔍 Verifying upload...")
        
        stats = self.index.describe_index_stats()
        print(f"📊 Final index stats:")
        print(f"   - Total vectors in index: {stats.total_vector_count}")
        print(f"   - Dimension: {stats.dimension if hasattr(stats, 'dimension') else 'N/A'}")
        
        if stats.namespaces:
            print(f"   - Namespaces:")
            for ns, ns_stats in stats.namespaces.items():
                ns_name = ns if ns else "(default)"
                print(f"     • {ns_name}: {ns_stats.vector_count} vectors")
    
    def save_pipeline_report(self):
        """Save pipeline execution report"""
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = self.config.OUTPUT_DIR / "embedding_pipeline_report.json"
        
        report = {
            'configuration': {
                'index_name': self.config.INDEX_NAME,
                'embedding_model': self.config.EMBEDDING_MODEL,
                'dimension': self.config.DIMENSION,
                'batch_size': self.config.BATCH_SIZE,
            },
            'statistics': self.stats,
            'success_rate': (
                (self.stats['uploaded_vectors'] / self.stats['total_chunks'] * 100)
                if self.stats['total_chunks'] > 0 else 0
            )
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Report saved to: {report_path}")
    
    def print_summary(self):
        """Print pipeline execution summary"""
        print("\n" + "="*70)
        print("EMBEDDING PIPELINE SUMMARY")
        print("="*70)
        print(f"Total chunks:        {self.stats['total_chunks']}")
        print(f"Successfully embedded: {self.stats['embedded_chunks']}")
        print(f"Uploaded to Pinecone:  {self.stats['uploaded_vectors']}")
        print(f"Failed chunks:       {self.stats['failed_chunks']}")
        
        if self.stats['errors']:
            print(f"\n⚠️  Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(self.stats['errors']) > 5:
                print(f"   ... and {len(self.stats['errors']) - 5} more")
        
        success_rate = (
            (self.stats['uploaded_vectors'] / self.stats['total_chunks'] * 100)
            if self.stats['total_chunks'] > 0 else 0
        )
        print(f"\nSuccess rate: {success_rate:.1f}%")
        print("="*70)
    
    def run(self):
        """Execute the complete pipeline"""
        print("="*70)
        print("="*70)
        
        try:
            # Step 1: Initialize clients
            self.initialize_clients()
            
            # Step 2: Setup index
            self.create_or_connect_index()
            
            # Step 3: Load chunks
            chunks = self.load_chunks()
            
            # Step 4: Process and upload
            self.process_and_upload_chunks(chunks)
            
            # Step 5: Verify
            self.verify_upload()
            
            # Step 6: Save report
            self.save_pipeline_report()
            
            # Step 7: Print summary
            self.print_summary()
            
            print("\n✅ PHASE C COMPLETE - READY FOR QUERYING")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise


# Query helper functions
def query_index(
    index,
    client: OpenAI,
    query_text: str,
    top_k: int = 5,
    filter_dict: Optional[Dict] = None,
    embedding_model: str = "text-embedding-3-large"
) -> Dict:
    """
    Query the Pinecone index with a text query.
    
    Args:
        index: Pinecone index object
        client: OpenAI client
        query_text: Text to search for
        top_k: Number of results to return
        filter_dict: Optional metadata filter
        embedding_model: Model to use for query embedding
    
    Returns:
        Query results with matches
    """
    # Create query embedding
    response = client.embeddings.create(
        input=[query_text],
        model=embedding_model
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True
    )
    
    return results


def print_query_results(results: Dict, show_text: bool = True):
    """Pretty print query results"""
    print(f"\n🔍 Found {len(results.matches)} results:\n")
    
    for i, match in enumerate(results.matches, 1):
        print(f"Result {i}:")
        print(f"  Score: {match.score:.4f}")
        print(f"  ID: {match.id}")
        
        if match.metadata:
            print(f"  Section: {match.metadata.get('section', 'N/A')}")
            print(f"  Pages: {match.metadata.get('pages', 'N/A')}")
            print(f"  Strategy: {match.metadata.get('strategy', 'N/A')}")
            print(f"  Word count: {match.metadata.get('word_count', 'N/A')}")
            
            if show_text and 'text' in match.metadata:
                text = match.metadata['text']
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"  Text preview: {preview}")
        
        print()


# Example filter templates
FILTER_EXAMPLES = {
    'by_page': {'pages': {'$eq': '[5]'}},  # Note: pages stored as JSON string
    'special_content': {'is_special': {'$eq': True}},
    'semantic_units': {'is_semantic': {'$eq': True}},
    'by_section': {'section': {'$eq': 'Introduction'}},
    'word_count_range': {
        '$and': [
            {'word_count': {'$gte': 100}},
            {'word_count': {'$lte': 500}}
        ]
    }
}


def main():
    """Main execution function"""
    # Create configuration
    config = Config()
    
    # Verify input file exists
    if not config.CHUNKS_PATH.exists():
        print(f"❌ Error: Chunks file not found at {config.CHUNKS_PATH}")
        print(f"   Please run Phase B chunking pipeline first.")
        return
    
    # Create and run pipeline
    pipeline = EmbeddingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()