#!/usr/bin/env python3
"""
Text Chunker Module - Cloud Composer Compatible
Contains HybridChunk dataclass and intelligent chunking logic
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
from google.cloud import storage
from urllib.parse import urlparse


@dataclass
class HybridChunk:
    """Represents an intelligently created hybrid chunk."""
    chunk_id: int
    content: str
    primary_strategy: str
    strategies_applied: List[str]
    section_path: str
    heading_hierarchy: List[str]
    page_numbers: List[int]
    content_types: List[str]
    is_special_content: bool
    is_semantic_unit: bool
    was_split: bool
    has_overlap: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    @property
    def full_text_with_context(self):
        """Content with hierarchical context"""
        if self.heading_hierarchy:
            context = " > ".join(self.heading_hierarchy)
            return f"[Context: {context}]\n\n{self.content}"
        return self.content


def chunk_blocks_intelligent(
    blocks: List[Dict],
    target_size: int = 1500,
    overlap: int = 150,
    max_size: int = 2500,
    min_size: int = 300,
    keep_tables_whole: bool = True,
    keep_code_whole: bool = True,
    respect_sections: bool = True,
    add_context: bool = True,
    skip_types: List[str] = None
) -> List[Dict]:
    """
    Create intelligent chunks from blocks list.
    Cloud Composer compatible - works with in-memory data.
    
    Returns list of dicts (not HybridChunk objects) for JSON serialization.
    """
    if skip_types is None:
        skip_types = ['figure']
    
    blocks.sort(key=lambda x: (x.get('page', 0), x.get('block_id', 0)))
    
    chunks = []
    chunk_id = 0
    
    heading_stack = []
    current_section = None
    current_section_content = []
    current_section_metadata = {
        'pages': set(),
        'block_ids': [],
        'content_types': set(),
        'has_special': False,
        'blocks': []
    }
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    def create_chunk(content: str, strategy: str, strategies_list: List[str],
                     is_special: bool = False, is_semantic: bool = False,
                     was_split: bool = False, has_overlap: bool = False):
        nonlocal chunk_id, chunks
        
        if not content.strip():
            return
        
        content_types = list(current_section_metadata['content_types'])
        if not content_types:
            content_types = ['text']
        
        metadata = {
            'char_count': len(content),
            'word_count': len(content.split()),
            'pages': sorted(list(current_section_metadata['pages'])),
            'block_ids': current_section_metadata['block_ids'].copy(),
            'num_blocks': len(current_section_metadata['block_ids']),
            'strategy': strategy,
            'strategies_applied': strategies_list,
            'target_size': target_size,
            'is_semantic_unit': is_semantic,
            'was_split': was_split,
            'has_overlap': has_overlap,
            'is_special_content': is_special,
            'size_category': 'optimal' if min_size <= len(content) <= max_size else 
                           ('small' if len(content) < min_size else 'large'),
            'chunking_method': 'hybrid_intelligent'
        }
        
        chunk = HybridChunk(
            chunk_id=chunk_id,
            content=content.strip(),
            primary_strategy=strategy,
            strategies_applied=strategies_list,
            section_path=current_section or 'Unknown',
            heading_hierarchy=[h['text'] for h in heading_stack],
            page_numbers=sorted(list(current_section_metadata['pages'])),
            content_types=content_types,
            is_special_content=is_special,
            is_semantic_unit=is_semantic,
            was_split=was_split,
            has_overlap=has_overlap,
            metadata=metadata
        )
        
        chunks.append(chunk.to_dict())
        chunk_id += 1
    
    def process_accumulated_content():
        nonlocal current_section_content, current_section_metadata
        
        if not current_section_content:
            return
        
        combined = '\n\n'.join(current_section_content).strip()
        if not combined:
            return
        
        content_size = len(combined)
        strategies = []
        
        if content_size <= max_size:
            strategies.append('semantic_section')
            if respect_sections:
                strategies.append('section_preservation')
            
            create_chunk(
                content=combined,
                strategy='semantic_section',
                strategies_list=strategies,
                is_semantic=True,
                is_special=current_section_metadata['has_special']
            )
        else:
            strategies.append('intelligent_split')
            if add_context:
                strategies.append('context_preservation')
            
            split_texts = text_splitter.split_text(combined)
            
            for i, split_text in enumerate(split_texts):
                chunk_strategies = strategies.copy()
                if i > 0 or i < len(split_texts) - 1:
                    chunk_strategies.append('overlap')
                
                create_chunk(
                    content=split_text,
                    strategy='intelligent_split',
                    strategies_list=chunk_strategies,
                    was_split=True,
                    has_overlap=len(split_texts) > 1
                )
        
        current_section_content = []
        current_section_metadata = {
            'pages': set(),
            'block_ids': [],
            'content_types': set(),
            'has_special': False,
            'blocks': []
        }
    
    # Process blocks
    for block in blocks:
        block_type = block.get('type', 'unknown')
        block_id = block.get('block_id')
        text = block.get('text', '').strip()
        section = block.get('section_path', '')
        page = block.get('page', 0)
        
        if block_type in skip_types or not text:
            continue
        
        if block_type == 'heading':
            level = block.get('heading_level', 1)
            
            if current_section_content and respect_sections:
                process_accumulated_content()
            
            heading_stack = [h for h in heading_stack if h['level'] < level]
            heading_stack.append({'level': level, 'text': text, 'section': section})
            current_section = section
            continue
        
        if block_type == 'table' and keep_tables_whole:
            process_accumulated_content()
            
            caption = block.get('caption', 'Table')
            table_content = f"## {caption}\n\n```\n{text}\n```"
            
            current_section_content = [table_content]
            current_section_metadata = {
                'pages': {page},
                'block_ids': [block_id],
                'content_types': {'table'},
                'has_special': True,
                'blocks': [block]
            }
            
            create_chunk(
                content=table_content,
                strategy='special_content_table',
                strategies_list=['special_content', 'table_preservation'],
                is_special=True,
                is_semantic=True
            )
            
            current_section_content = []
            current_section_metadata = {
                'pages': set(),
                'block_ids': [],
                'content_types': set(),
                'has_special': False,
                'blocks': []
            }
            continue
        
        if block_type == 'code' and keep_code_whole:
            process_accumulated_content()
            
            code_content = f"```\n{text}\n```"
            
            current_section_content = [code_content]
            current_section_metadata = {
                'pages': {page},
                'block_ids': [block_id],
                'content_types': {'code'},
                'has_special': True,
                'blocks': [block]
            }
            
            create_chunk(
                content=code_content,
                strategy='special_content_code',
                strategies_list=['special_content', 'code_preservation'],
                is_special=True,
                is_semantic=True
            )
            
            current_section_content = []
            current_section_metadata = {
                'pages': set(),
                'block_ids': [],
                'content_types': set(),
                'has_special': False,
                'blocks': []
            }
            continue
        
        # Regular content
        formatted_text = text
        if block_type == 'equation':
            formatted_text = f"$$\n{text}\n$$"
        elif block_type == 'list':
            formatted_text = text
        elif block_type == 'table':
            formatted_text = f"```\n{text}\n```"
            current_section_metadata['has_special'] = True
        
        current_size = sum(len(c) for c in current_section_content)
        
        if current_size + len(formatted_text) > max_size and current_section_content:
            process_accumulated_content()
        
        current_section_content.append(formatted_text)
        current_section_metadata['pages'].add(page)
        current_section_metadata['block_ids'].append(block_id)
        current_section_metadata['content_types'].add(block_type)
        current_section_metadata['blocks'].append(block)
    
    process_accumulated_content()
    
    print(f"✅ Created {len(chunks)} intelligent chunks")
    return chunks


def chunk_from_gcs(
    blocks_gs_url: str,
    output_gs_url: str,
    target_size: int = 1500,
    overlap: int = 150,
    max_size: int = 2500
) -> Dict:
    """
    Cloud Composer wrapper: Load blocks from GCS, chunk, save to GCS.
    
    Args:
        blocks_gs_url: Input blocks JSONL (gs://...)
        output_gs_url: Output chunks JSONL (gs://...)
        target_size: Target chunk size
        overlap: Overlap size
        max_size: Max chunk size
    
    Returns:
        Statistics dict
    """
    # Load blocks from GCS
    u = urlparse(blocks_gs_url)
    content = storage.Client().bucket(u.netloc).blob(u.path.lstrip("/")).download_as_text()
    blocks = [json.loads(line) for line in content.splitlines() if line.strip()]
    
    # Chunk
    chunks = chunk_blocks_intelligent(
        blocks=blocks,
        target_size=target_size,
        overlap=overlap,
        max_size=max_size
    )
    
    # Save to GCS
    tmp = "/tmp/chunks.jsonl"
    with open(tmp, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    u_out = urlparse(output_gs_url)
    storage.Client().bucket(u_out.netloc).blob(u_out.path.lstrip("/")).upload_from_filename(tmp)
    os.remove(tmp)
    
    return {
        'source': blocks_gs_url,
        'output': output_gs_url,
        'chunks_count': len(chunks)
    }

def prepare_chunks_for_embedding(chunks: List[Dict]) -> List[Dict]:
    """
    Prepare chunks in format ready for embedding.
    Adds context and ensures proper structure.
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