#!/usr/bin/env python3
"""
Aurelia RAG Pipeline - Modular Architecture
Main orchestration DAG for PDF → Parse → Chunk → Embed workflow

Structure:
- Phase A: PDF Parsing (utils/pdf_parser.py)
- Phase B: Intelligent Chunking (utils/text_chunker.py)
- Phase C: Vector Embedding (utils/vector_embedder.py)
"""

from datetime import datetime, timedelta
from urllib.parse import urlparse
import json
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from google.cloud import storage


# ============================================================================
# PHASE A: PDF PARSING
# ============================================================================

def list_input_pdfs(**context):
    """List all PDFs in input bucket"""
    input_prefix = Variable.get("PDF_INPUT_PREFIX")
    
    u = urlparse(input_prefix)
    blobs = storage.Client().list_blobs(u.netloc, prefix=u.path.lstrip("/"))
    pdfs = [f"gs://{u.netloc}/{b.name}" for b in blobs if b.name.endswith('.pdf')]
    
    if not pdfs:
        print("⚠️  No PDFs found")
        return []
    
    print(f"✓ Found {len(pdfs)} PDFs")
    context['ti'].xcom_push(key='pdf_list', value=pdfs)
    return pdfs


def parse_pdf_task(**context):
    """Parse PDF using EnhancedPyMuPDFParser"""
    from utils.pdf_parser import parse_pdf_from_gcs
    
    pdfs = context['ti'].xcom_pull(key='pdf_list', task_ids='phase_a.list_pdfs')
    if not pdfs:
        return None
    
    pdf_url = pdfs[0]
    
    # Define output location
    output_prefix = Variable.get("PARSED_OUTPUT_PREFIX")
    filename = os.path.basename(pdf_url).replace('.pdf', '_blocks.jsonl')
    output_url = f"{output_prefix.rstrip('/')}/{filename}"
    
    # Parse
    print(f"📄 Parsing: {pdf_url}")
    result = parse_pdf_from_gcs(
        pdf_gs_url=pdf_url,
        output_gs_url=output_url,
        config=None
    )
    
    print(f"✓ Parsed {result['blocks_count']} blocks")
    
    context['ti'].xcom_push(key='blocks_file', value=output_url)
    context['ti'].xcom_push(key='blocks_count', value=result['blocks_count'])
    
    return output_url


# ============================================================================
# PHASE B: INTELLIGENT CHUNKING
# ============================================================================

def chunk_document_task(**context):
    """Create intelligent chunks from blocks"""
    from utils.text_chunker import chunk_from_gcs
    
    blocks_file = context['ti'].xcom_pull(key='blocks_file', task_ids='phase_a.parse_pdf')
    if not blocks_file:
        return None
    
    # Define output location
    output_prefix = Variable.get("CHUNKS_OUTPUT_PREFIX")
    filename = os.path.basename(blocks_file).replace('_blocks.jsonl', '_chunks.jsonl')
    output_url = f"{output_prefix.rstrip('/')}/{filename}"
    
    # Chunk
    print(f"📚 Chunking: {blocks_file}")
    result = chunk_from_gcs(
        blocks_gs_url=blocks_file,
        output_gs_url=output_url,
        target_size=int(Variable.get('CHUNK_TARGET_SIZE', default_var='1500')),
        overlap=int(Variable.get('CHUNK_OVERLAP', default_var='150')),
        max_size=int(Variable.get('CHUNK_MAX_SIZE', default_var='2500'))
    )
    
    print(f"✓ Created {result['chunks_count']} chunks")
    
    context['ti'].xcom_push(key='chunks_file', value=output_url)
    context['ti'].xcom_push(key='chunks_count', value=result['chunks_count'])
    
    return output_url


def prepare_for_embedding_task(**context):
    """Prepare chunks in embedding-ready format"""
    from google.cloud import storage
    from urllib.parse import urlparse
    from utils.text_chunker import prepare_chunks_for_embedding
    
    chunks_file = context['ti'].xcom_pull(key='chunks_file', task_ids='phase_b.chunk_document')
    if not chunks_file:
        return None
    
    # Load chunks
    u = urlparse(chunks_file)
    content = storage.Client().bucket(u.netloc).blob(u.path.lstrip("/")).download_as_text()
    chunks = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    # Prepare for embedding
    embedding_chunks = prepare_chunks_for_embedding(chunks)
    
    # Save
    output_prefix = Variable.get("CHUNKS_OUTPUT_PREFIX")
    output_url = f"{output_prefix.rstrip('/')}/chunks_for_embedding.jsonl"
    
    tmp = "/tmp/embedding_chunks.jsonl"
    with open(tmp, 'w', encoding='utf-8') as f:
        for chunk in embedding_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    u_out = urlparse(output_url)
    storage.Client().bucket(u_out.netloc).blob(u_out.path.lstrip("/")).upload_from_filename(tmp)
    os.remove(tmp)
    
    print(f"✓ Prepared {len(embedding_chunks)} chunks for embedding")
    
    context['ti'].xcom_push(key='embedding_file', value=output_url)
    context['ti'].xcom_push(key='embedding_count', value=len(embedding_chunks))
    
    return output_url


# ============================================================================
# PHASE C: VECTOR EMBEDDING
# ============================================================================

def create_embeddings_task(**context):
    """Create embeddings and upload to Pinecone"""
    from utils.vector_embedder import embed_chunks_to_pinecone
    
    chunks_file = context['ti'].xcom_pull(key='embedding_file', task_ids='phase_b.prepare_embedding')
    if not chunks_file:
        return None
    
    index_name = Variable.get('PINECONE_INDEX_NAME', default_var='fintbx-embeddings')
    embedding_model = Variable.get('EMBEDDING_MODEL', default_var='text-embedding-3-large')
    
    print(f"🚀 Embedding to Pinecone index: {index_name}")
    
    result = embed_chunks_to_pinecone(
        chunks_gs_url=chunks_file,
        index_name=index_name,
        embedding_model=embedding_model,
        dimension=3072,
        batch_size=100
    )
    
    print(f"✓ Embedded {result['embedded']} chunks")
    print(f"✓ Success rate: {result['success_rate']:.1f}%")
    
    context['ti'].xcom_push(key='embedded_count', value=result['embedded'])
    context['ti'].xcom_push(key='total_in_index', value=result['total_in_index'])
    
    return result


# ============================================================================
# REPORTING
# ============================================================================

def print_summary(**context):
    """Print pipeline summary"""
    blocks_count = context['ti'].xcom_pull(key='blocks_count', task_ids='phase_a.parse_pdf')
    chunks_count = context['ti'].xcom_pull(key='chunks_count', task_ids='phase_b.chunk_document')
    embedded_count = context['ti'].xcom_pull(key='embedded_count', task_ids='phase_c.create_embeddings')
    total_in_index = context['ti'].xcom_pull(key='total_in_index', task_ids='phase_c.create_embeddings')
    
    print("="*80)
    print("RAG PIPELINE COMPLETE")
    print("="*80)
    print(f"Phase A - PDF Parsing:    {blocks_count or 0} blocks extracted")
    print(f"Phase B - Chunking:       {chunks_count or 0} chunks created")
    print(f"Phase C - Embedding:      {embedded_count or 0} vectors uploaded")
    print(f"Total in Pinecone Index:  {total_in_index or 0}")
    print("="*80)
    print("✅ READY FOR QUERYING")
    print("="*80)


# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = {
    'owner': 'aurelia',
    'depends_on_past': False,
    'email_on_failure': False,  # Disable to avoid SMTP errors
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
}

with DAG(
    dag_id='aurelia_rag_pipeline',
    default_args=default_args,
    description='Complete RAG Pipeline: PDF → Parse → Chunk → Embed',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['aurelia', 'rag', 'pdf', 'pinecone'],
) as dag:
    
    # Phase A: PDF Parsing
    with TaskGroup('phase_a', tooltip='PDF Parsing') as phase_a:
        list_pdfs = PythonOperator(
            task_id='list_pdfs',
            python_callable=list_input_pdfs,
        )
        
        parse_pdf = PythonOperator(
            task_id='parse_pdf',
            python_callable=parse_pdf_task,
            execution_timeout=timedelta(hours=3),
        )
        
        list_pdfs >> parse_pdf
    
    # Phase B: Chunking
    with TaskGroup('phase_b', tooltip='Intelligent Chunking') as phase_b:
        chunk_doc = PythonOperator(
            task_id='chunk_document',
            python_callable=chunk_document_task,
            execution_timeout=timedelta(hours=3),
        )
        
        prep_embed = PythonOperator(
            task_id='prepare_embedding',
            python_callable=prepare_for_embedding_task,
            execution_timeout=timedelta(hours=2),
        )
        
        chunk_doc >> prep_embed
    
    # Phase C: Embedding
    with TaskGroup('phase_c', tooltip='Vector Embedding') as phase_c:
        create_embed = PythonOperator(
            task_id='create_embeddings',
            python_callable=create_embeddings_task,
            execution_timeout=timedelta(hours=2),
        )
    
    # Summary
    summary = PythonOperator(
        task_id='print_summary',
        python_callable=print_summary,
    )
    
    # Pipeline flow
    phase_a >> phase_b >> phase_c >> summary