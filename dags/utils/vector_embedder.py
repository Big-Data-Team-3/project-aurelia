#!/usr/bin/env python3
"""
Enhanced Vector Embedder Module - Cloud Composer Compatible
Full-featured embedding with retry logic, metadata limits, and robust error handling
"""

import json
import os
import time
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from google.cloud import storage
from urllib.parse import urlparse


# Configuration constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
PINECONE_TEXT_LIMIT = 40000  # Pinecone metadata text field limit
PINECONE_METADATA_LIMIT = 500  # Safe limit for string fields
PINECONE_BATCH_SIZE = 100  # Vectors per upsert batch


def _delete_index_if_exists(pc: Pinecone, index_name: str):
    """Delete Pinecone index if it exists"""
    existing = [idx.name for idx in pc.list_indexes()]
    
    if index_name in existing:
        print(f"🗑️  Deleting existing index: {index_name}")
        pc.delete_index(index_name)
        print(f"   Waiting for deletion to complete...")
        time.sleep(5)  # Wait for deletion to propagate
        print(f"✅ Index deleted successfully")
    else:
        print(f"ℹ️  Index '{index_name}' does not exist, skipping deletion")


def delete_pinecone_index(index_name: str) -> Dict:
    """
    Standalone function to delete a Pinecone index.
    Can be called as a separate DAG task.
    
    Args:
        index_name: Name of the index to delete
    
    Returns:
        Status dict
    """
    print(f"🗑️  Deleting Pinecone index: {index_name}")
    
    # Get API key
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY must be set in Composer environment")
    
    # Initialize client
    pc = Pinecone(api_key=pinecone_key)
    
    # Delete index
    _delete_index_if_exists(pc, index_name)
    
    return {
        'index_name': index_name,
        'status': 'deleted',
        'timestamp': time.time()
    }


def embed_chunks_to_pinecone(
    chunks_gs_url: str,
    index_name: str,
    embedding_model: str = 'text-embedding-3-large',
    dimension: int = 3072,
    batch_size: int = 100,
    recreate_index: bool = False
) -> Dict:
    """
    Cloud Composer wrapper: Load chunks from GCS, embed with retry logic, upload to Pinecone.
    
    Features:
    - Automatic retry with exponential backoff
    - Metadata length truncation for Pinecone limits
    - Sub-batching for reliable uploads
    - Comprehensive error tracking
    
    Args:
        chunks_gs_url: Input chunks JSONL (gs://...)
        index_name: Pinecone index name
        embedding_model: OpenAI model name
        dimension: Vector dimension
        batch_size: Batch size for embedding creation
        recreate_index: If True, delete and recreate index (default: False)
    
    Returns:
        Statistics dict with detailed metrics
    """
    print(f"🚀 Starting embedding pipeline")
    print(f"   Model: {embedding_model}")
    print(f"   Batch size: {batch_size}")
    print(f"   Recreate index: {recreate_index}")
    
    # Get API keys from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    
    if not openai_key or not pinecone_key:
        raise ValueError("OPENAI_API_KEY and PINECONE_API_KEY must be set in Composer environment")
    
    # Initialize clients
    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    
    # Delete index if recreate_index=True
    if recreate_index:
        _delete_index_if_exists(pc, index_name)
    
    # Setup index with retry
    print(f"📊 Setting up Pinecone index: {index_name}")
    index = _setup_pinecone_index(pc, index_name, dimension)
    
    # Load chunks from GCS
    print(f"📂 Loading chunks from GCS...")
    chunks = _load_chunks_from_gcs(chunks_gs_url)
    print(f"   Loaded {len(chunks)} chunks")
    
    # Process with full pipeline
    results = _process_chunks_with_retry(
        client=client,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        batch_size=batch_size
    )
    
    # Get final stats
    stats = index.describe_index_stats()
    
    final_results = {
        'source': chunks_gs_url,
        'index_name': index_name,
        'chunks_processed': len(chunks),
        'embedded': results['embedded'],
        'failed': results['failed'],
        'total_in_index': stats.total_vector_count,
        'success_rate': (results['embedded'] / len(chunks) * 100) if len(chunks) > 0 else 0,
        'errors': results['errors'][:10]  # First 10 errors
    }
    
    print(f"\n✅ Embedding complete!")
    print(f"   Embedded: {final_results['embedded']}/{len(chunks)}")
    print(f"   Success rate: {final_results['success_rate']:.1f}%")
    print(f"   Total in index: {final_results['total_in_index']}")
    
    return final_results


def _setup_pinecone_index(pc: Pinecone, index_name: str, dimension: int):
    """Setup or connect to Pinecone index with retry logic"""
    existing = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing:
        print(f"   Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print(f"   Waiting for index to be ready...")
        time.sleep(10)
    else:
        print(f"   Index already exists, connecting...")
    
    index = pc.Index(index_name)
    
    # Verify connection
    try:
        stats = index.describe_index_stats()
        print(f"   Current vectors: {stats.total_vector_count}")
    except Exception as e:
        print(f"   Warning: Could not get index stats: {e}")
    
    return index


def _load_chunks_from_gcs(chunks_gs_url: str) -> List[Dict]:
    """Load chunks from GCS with error handling"""
    try:
        u = urlparse(chunks_gs_url)
        content = storage.Client().bucket(u.netloc).blob(u.path.lstrip("/")).download_as_text()
        chunks = [json.loads(line) for line in content.splitlines() if line.strip()]
        return chunks
    except Exception as e:
        raise ValueError(f"Failed to load chunks from GCS: {e}")


def _process_chunks_with_retry(
    client: OpenAI,
    index,
    chunks: List[Dict],
    embedding_model: str,
    batch_size: int
) -> Dict:
    """Process chunks with retry logic and error tracking"""
    
    embedded = 0
    failed = 0
    errors = []
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"\n🔄 Processing {total_batches} batches...")
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            # Extract texts with context
            texts = _prepare_texts_for_embedding(batch)
            
            # Create embeddings with retry
            embeddings = _create_embeddings_with_retry(
                client=client,
                texts=texts,
                model=embedding_model
            )
            
            if not embeddings:
                failed += len(batch)
                errors.append(f"Batch {batch_num}: Failed to create embeddings")
                continue
            
            # Prepare vectors with metadata limits
            vectors = _prepare_vectors_for_pinecone(batch, embeddings, i)
            
            # Upload with sub-batching
            uploaded = _upload_vectors_with_subbatching(index, vectors, batch_num)
            
            embedded += uploaded
            
            if batch_num % 5 == 0:
                print(f"   Progress: {batch_num}/{total_batches} batches ({embedded} vectors)")
            
        except Exception as e:
            error_msg = f"Batch {batch_num}: {str(e)}"
            errors.append(error_msg)
            failed += len(batch)
            print(f"   ❌ {error_msg}")
            continue
    
    return {
        'embedded': embedded,
        'failed': failed,
        'errors': errors
    }


def _prepare_texts_for_embedding(batch: List[Dict]) -> List[str]:
    """Prepare texts with context for embedding"""
    texts = []
    
    for chunk in batch:
        # Handle both formats
        text = chunk.get('content') or chunk.get('text', '')
        
        # Add hierarchical context if available
        if 'heading_hierarchy' in chunk and chunk['heading_hierarchy']:
            context_str = ' > '.join(chunk['heading_hierarchy'])
            text = f"[Context: {context_str}]\n\n{text}"
        
        texts.append(text)
    
    return texts


def _create_embeddings_with_retry(
    client: OpenAI,
    texts: List[str],
    model: str
) -> List[List[float]]:
    """Create embeddings with exponential backoff retry"""
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                input=texts,
                model=model
            )
            return [item.embedding for item in response.data]
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"   ⚠️ Retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"   ❌ Failed after {MAX_RETRIES} attempts: {e}")
                raise e
    
    return []


def _prepare_vectors_for_pinecone(
    batch: List[Dict],
    embeddings: List[List[float]],
    batch_start_idx: int
) -> List[Dict]:
    """Prepare vectors with Pinecone-compatible metadata"""
    vectors = []
    
    for j, chunk in enumerate(batch):
        metadata = chunk.get('metadata', {})
        
        # Clean and truncate metadata for Pinecone limits
        clean_meta = _prepare_metadata_for_pinecone(chunk, metadata)
        
        # Generate ID
        chunk_id = str(chunk.get('chunk_id') or chunk.get('id', f"chunk_{batch_start_idx}_{j}"))
        
        vectors.append({
            'id': chunk_id,
            'values': embeddings[j],
            'metadata': clean_meta
        })
    
    return vectors


def _prepare_metadata_for_pinecone(chunk: Dict, metadata: Dict) -> Dict:
    """
    Prepare metadata for Pinecone with proper limits and type handling.
    
    Pinecone limits:
    - Text field: 40,000 characters
    - String fields: Keep under 500 chars for safety
    - Supported types: str, int, float, bool, list of str
    """
    clean_meta = {}
    
    # Add text with strict limit
    text = chunk.get('content') or chunk.get('text', '')
    if text:
        clean_meta['text'] = text[:PINECONE_TEXT_LIMIT]
    
    # Process metadata fields
    for key, value in metadata.items():
        if key == 'text':
            continue  # Already handled above
        
        # Handle different types
        if isinstance(value, (list, dict)):
            # Convert complex types to JSON string and truncate
            json_str = json.dumps(value)
            clean_meta[key] = json_str[:PINECONE_METADATA_LIMIT]
        
        elif isinstance(value, str):
            # Truncate strings
            clean_meta[key] = value[:PINECONE_METADATA_LIMIT]
        
        elif isinstance(value, (int, float, bool, type(None))):
            # Direct assignment for simple types
            clean_meta[key] = value
        
        else:
            # Convert unknown types to string and truncate
            clean_meta[key] = str(value)[:PINECONE_METADATA_LIMIT]
    
    # Add essential fields if missing
    if 'chunk_id' not in clean_meta:
        clean_meta['chunk_id'] = str(chunk.get('chunk_id') or chunk.get('id', 0))
    
    if 'strategy' not in clean_meta:
        clean_meta['strategy'] = chunk.get('primary_strategy', 'unknown')
    
    if 'section' not in clean_meta:
        section = chunk.get('section_path', '')
        clean_meta['section'] = section[:PINECONE_METADATA_LIMIT]
    
    return clean_meta


def _upload_vectors_with_subbatching(
    index,
    vectors: List[Dict],
    batch_num: int
) -> int:
    """Upload vectors with sub-batching for reliability"""
    uploaded = 0
    
    # Split into sub-batches
    for k in range(0, len(vectors), PINECONE_BATCH_SIZE):
        sub_batch = vectors[k:k + PINECONE_BATCH_SIZE]
        
        try:
            index.upsert(vectors=sub_batch)
            uploaded += len(sub_batch)
            
        except Exception as e:
            print(f"   ⚠️ Sub-batch upload failed in batch {batch_num}: {e}")
            # Continue with next sub-batch instead of failing entire batch
            continue
    
    return uploaded


def prepare_chunks_for_embedding(chunks: List[Dict]) -> List[Dict]:
    """
    Prepare chunks in format ready for embedding.
    Adds context and ensures proper structure.
    
    This function is for compatibility and data transformation.
    The actual embedding logic handles context addition internally.
    """
    embedding_ready = []
    
    for chunk in chunks:
        text = chunk.get('content') or chunk.get('text', '')
        
        # Add hierarchical context if available
        if chunk.get('heading_hierarchy'):
            context_str = ' > '.join(chunk['heading_hierarchy'])
            text = f"[Context: {context_str}]\n\n{text}"
        
        embedding_ready.append({
            'id': chunk.get('chunk_id', 0),
            'text': text,
            'metadata': {
                'chunk_id': chunk.get('chunk_id', 0),
                'strategy': chunk.get('primary_strategy', 'unknown'),
                'section': chunk.get('section_path', ''),
                'pages': chunk.get('page_numbers', []),
                'is_special': chunk.get('is_special_content', False),
                'is_semantic': chunk.get('is_semantic_unit', False),
                'char_count': len(text),
                'word_count': len(text.split())
            }
        })
    
    return embedding_ready


# ============================================================================
# QUERY HELPER FUNCTIONS (for future use)
# ============================================================================

def query_index(
    index,
    client: OpenAI,
    query_text: str,
    top_k: int = 5,
    filter_dict: Dict = None,
    embedding_model: str = "text-embedding-3-large"
) -> Dict:
    """
    Query the Pinecone index with a text query.
    
    Args:
        index: Pinecone index object
        client: OpenAI client
        query_text: Text to search for
        top_k: Number of results
        filter_dict: Optional metadata filter
        embedding_model: Model for query embedding
    
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