#!/usr/bin/env python3
"""
Phase B - Hybrid Intelligent Chunking Pipeline
Creates intelligent chunks using multiple strategies from cleaned blocks
"""

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict


@dataclass
class HybridChunk:
    """
    Represents an intelligently created hybrid chunk.
    Combines multiple strategies for optimal chunking.
    """
    chunk_id: int
    content: str
    primary_strategy: str  # Which strategy created this chunk
    strategies_applied: List[str]  # All strategies used
    
    # Context information
    section_path: str
    heading_hierarchy: List[str]
    page_numbers: List[int]
    
    # Content classification
    content_types: List[str]
    is_special_content: bool
    
    # Chunking metadata
    is_semantic_unit: bool  # True if kept whole for semantic reasons
    was_split: bool  # True if larger unit was split
    has_overlap: bool  # True if overlaps with neighbors
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    def __len__(self):
        return len(self.content)
    
    @property
    def full_text_with_context(self):
        """Content with hierarchical context"""
        if self.heading_hierarchy:
            context = " > ".join(self.heading_hierarchy)
            return f"[Context: {context}]\n\n{self.content}"
        return self.content


def chunk_hybrid_intelligent(
    jsonl_path: str,
    target_size: int = 1500,
    overlap: int = 150,
    max_size: int = 2500,
    min_size: int = 300,
    keep_tables_whole: bool = True,
    keep_code_whole: bool = True,
    respect_sections: bool = True,
    add_context: bool = True,
    skip_types: List[str] = ['figure']
) -> List[HybridChunk]:
    """
    Create hybrid intelligent chunks using multiple strategies:
    
    Strategy Priority:
    1. Preserve special content (tables, code, equations) whole
    2. Keep semantic sections together when possible
    3. Add hierarchical context to all chunks
    4. Apply size-based splitting only when necessary
    5. Add overlap for split chunks
    6. Track pages for citation
    
    Args:
        jsonl_path: Path to cleaned JSONL file
        target_size: Target chunk size in characters
        overlap: Overlap size for split chunks
        max_size: Maximum chunk size before forcing split
        min_size: Minimum chunk size (merge if smaller)
        keep_tables_whole: Preserve table integrity
        keep_code_whole: Preserve code block integrity
        respect_sections: Try to keep sections together
        add_context: Add hierarchical breadcrumbs
        skip_types: Block types to skip
    
    Returns:
        List of HybridChunk objects
    """
    
    print(f"🧠 Starting Hybrid Intelligent Chunking...")
    print(f"📏 Target size: {target_size} chars (range: {min_size}-{max_size})")
    print(f"🔄 Overlap: {overlap} chars")
    print(f"⚙️  Strategies:")
    print(f"   ✓ Special content preservation: {keep_tables_whole or keep_code_whole}")
    print(f"   ✓ Semantic sections: {respect_sections}")
    print(f"   ✓ Hierarchical context: {add_context}")
    print(f"   ✓ Intelligent splitting: Always")
    print()
    
    # Load blocks
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        blocks = [json.loads(line) for line in f if line.strip()]
    
    blocks.sort(key=lambda x: (x.get('page', 0), x.get('block_id', 0)))
    
    chunks = []
    chunk_id = 0
    
    # State tracking
    heading_stack = []  # Current heading hierarchy
    current_section = None
    current_section_content = []
    current_section_metadata = {
        'pages': set(),
        'block_ids': [],
        'content_types': set(),
        'has_special': False,
        'blocks': []
    }
    
    # Text splitter for when we need it
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )
    
    def create_chunk(
        content: str,
        strategy: str,
        strategies_list: List[str],
        is_special: bool = False,
        is_semantic: bool = False,
        was_split: bool = False,
        has_overlap: bool = False
    ):
        """Helper to create a hybrid chunk"""
        nonlocal chunk_id, chunks
        
        if not content.strip():
            return
        
        # Build context
        context_header = ""
        if add_context and heading_stack:
            hierarchy = [h['text'] for h in heading_stack]
            context_header = f"[Context: {' > '.join(hierarchy)}]\n\n"
        
        # Determine content types
        content_types = list(current_section_metadata['content_types'])
        if not content_types:
            content_types = ['text']
        
        # Create metadata
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
            'source': 'fintbx.pdf',
            'chunking_method': 'hybrid_intelligent'
        }
        
        # Create chunk
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
        
        chunks.append(chunk)
        chunk_id += 1
    
    def process_accumulated_content():
        """Process accumulated section content with intelligent strategy"""
        nonlocal current_section_content, current_section_metadata
        
        if not current_section_content:
            return
        
        combined = '\n\n'.join(current_section_content).strip()
        if not combined:
            return
        
        content_size = len(combined)
        strategies = []
        
        # STRATEGY 1: Small enough - keep as is (semantic)
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
        
        # STRATEGY 2: Too large - intelligent split
        else:
            strategies.append('intelligent_split')
            if add_context:
                strategies.append('context_preservation')
            
            # Use recursive splitter
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
        
        # Reset
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
        
        # Skip certain types
        if block_type in skip_types or not text:
            continue
        
        # HEADING: Update hierarchy and potentially flush
        if block_type == 'heading':
            level = block.get('heading_level', 1)
            
            # Flush accumulated content before hierarchy change
            if current_section_content and respect_sections:
                process_accumulated_content()
            
            # Update heading stack
            heading_stack = [h for h in heading_stack if h['level'] < level]
            heading_stack.append({
                'level': level,
                'text': text,
                'section': section
            })
            
            current_section = section
            continue
        
        # TABLE: Keep whole as special content
        if block_type == 'table' and keep_tables_whole:
            # Flush current content first
            process_accumulated_content()
            
            caption = block.get('caption', 'Table')
            table_content = f"## {caption}\n\n```\n{text}\n```"
            
            # Create standalone table chunk
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
            
            # Reset
            current_section_content = []
            current_section_metadata = {
                'pages': set(),
                'block_ids': [],
                'content_types': set(),
                'has_special': False,
                'blocks': []
            }
            continue
        
        # CODE: Keep whole as special content
        if block_type == 'code' and keep_code_whole:
            # Flush current content first
            process_accumulated_content()
            
            code_content = f"```\n{text}\n```"
            
            # Create standalone code chunk
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
            
            # Reset
            current_section_content = []
            current_section_metadata = {
                'pages': set(),
                'block_ids': [],
                'content_types': set(),
                'has_special': False,
                'blocks': []
            }
            continue
        
        # REGULAR CONTENT: Accumulate
        formatted_text = ""
        
        if block_type == 'equation':
            formatted_text = f"$$\n{text}\n$$"
        elif block_type == 'list':
            formatted_text = text
        elif block_type == 'table':
            # Table but not keeping whole
            formatted_text = f"```\n{text}\n```"
            current_section_metadata['has_special'] = True
        else:
            formatted_text = text
        
        # Check if adding would exceed max size
        current_size = sum(len(c) for c in current_section_content)
        
        if current_size + len(formatted_text) > max_size and current_section_content:
            # Process what we have
            process_accumulated_content()
        
        # Add to accumulation
        current_section_content.append(formatted_text)
        current_section_metadata['pages'].add(page)
        current_section_metadata['block_ids'].append(block_id)
        current_section_metadata['content_types'].add(block_type)
        current_section_metadata['blocks'].append(block)
    
    # Process final accumulated content
    process_accumulated_content()
    
    # Statistics
    strategies_used = {}
    for chunk in chunks:
        strategies_used[chunk.primary_strategy] = strategies_used.get(chunk.primary_strategy, 0) + 1
    
    print(f" Created {len(chunks)} hybrid intelligent chunks")
    print(f"\n Strategies Used:")
    for strategy, count in sorted(strategies_used.items(), key=lambda x: x[1], reverse=True):
        print(f"   {strategy:25s}: {count:3d} chunks")
    
    special_count = sum(1 for c in chunks if c.is_special_content)
    semantic_count = sum(1 for c in chunks if c.is_semantic_unit)
    split_count = sum(1 for c in chunks if c.was_split)
    
    print(f"\n Chunk Characteristics:")
    print(f"   Special content:     {special_count} ({special_count/len(chunks)*100:.1f}%)")
    print(f"   Semantic units:      {semantic_count} ({semantic_count/len(chunks)*100:.1f}%)")
    print(f"   Split chunks:        {split_count} ({split_count/len(chunks)*100:.1f}%)")
    
    return chunks


def analyze_hybrid_chunks(chunks: List[HybridChunk]) -> pd.DataFrame:
    """Generate comprehensive statistics about hybrid chunks"""
    
    stats = []
    for chunk in chunks:
        stats.append({
            'chunk_id': chunk.chunk_id,
            'strategy': chunk.primary_strategy,
            'num_strategies': len(chunk.strategies_applied),
            'is_special': chunk.is_special_content,
            'is_semantic': chunk.is_semantic_unit,
            'was_split': chunk.was_split,
            'has_overlap': chunk.has_overlap,
            'char_count': len(chunk.content),
            'word_count': len(chunk.content.split()),
            'size_category': chunk.metadata.get('size_category', 'unknown'),
            'pages': ', '.join(map(str, chunk.page_numbers)),
            'num_pages': len(chunk.page_numbers),
            'hierarchy_depth': len(chunk.heading_hierarchy),
            'content_types': ', '.join(chunk.content_types),
            'section': chunk.section_path[:40] + '...' if len(chunk.section_path) > 40 else chunk.section_path
        })
    
    df = pd.DataFrame(stats)
    
    print("\n HYBRID INTELLIGENT CHUNK STATISTICS")
    print("="*70)
    print(f"Total Chunks: {len(chunks)}")
    
    print(f"\n Primary Strategy Distribution:")
    strategy_dist = df['strategy'].value_counts()
    for strategy, count in strategy_dist.items():
        pct = (count / len(chunks)) * 100
        print(f"  {strategy:30s}: {count:3d} chunks ({pct:5.1f}%)")
    
    print(f"\n Size Distribution:")
    print(f"  Average Characters: {df['char_count'].mean():.0f}")
    print(f"  Median Characters:  {df['char_count'].median():.0f}")
    print(f"  Std Dev:            {df['char_count'].std():.0f}")
    print(f"  Min:                {df['char_count'].min()}")
    print(f"  Max:                {df['char_count'].max()}")
    
    print(f"\nSize Categories:")
    size_cats = df['size_category'].value_counts()
    for cat, count in size_cats.items():
        print(f"  {cat.capitalize():10s}: {count:3d} chunks")
    
    print(f"\nChunk Characteristics:")
    print(f"  Special Content:    {df['is_special'].sum():3d} ({df['is_special'].sum()/len(chunks)*100:.1f}%)")
    print(f"  Semantic Units:     {df['is_semantic'].sum():3d} ({df['is_semantic'].sum()/len(chunks)*100:.1f}%)")
    print(f"  Split Chunks:       {df['was_split'].sum():3d} ({df['was_split'].sum()/len(chunks)*100:.1f}%)")
    print(f"  With Overlap:       {df['has_overlap'].sum():3d} ({df['has_overlap'].sum()/len(chunks)*100:.1f}%)")
    
    print(f"\n📖 Context & Structure:")
    print(f"  Avg Hierarchy Depth: {df['hierarchy_depth'].mean():.1f}")
    print(f"  Avg Pages per Chunk: {df['num_pages'].mean():.1f}")
    print(f"  Multi-page Chunks:   {sum(df['num_pages'] > 1)}")
    
    print(f"\n Top 5 Largest Chunks:")
    print(df.nlargest(5, 'char_count')[['chunk_id', 'strategy', 'is_special', 'char_count', 'section']])
    
    return df


def preview_hybrid_chunks(chunks: List[HybridChunk], num_samples: int = 5):
    """Display diverse sample chunks showing different strategies"""
    
    print("\n📖 SAMPLE HYBRID INTELLIGENT CHUNKS")
    print("="*70)
    
    # Get diverse samples by strategy
    strategies = {}
    for chunk in chunks:
        strategy = chunk.primary_strategy
        if strategy not in strategies:
            strategies[strategy] = []
        strategies[strategy].append(chunk)
    
    samples = []
    for strategy, strategy_chunks in strategies.items():
        if strategy_chunks:
            samples.append(strategy_chunks[0])
    
    # Fill remaining with random chunks
    while len(samples) < num_samples and len(samples) < len(chunks):
        for chunk in chunks:
            if chunk not in samples:
                samples.append(chunk)
                break
    
    for i, chunk in enumerate(samples[:num_samples], 1):
        print(f"\n{'─'*70}")
        print(f"SAMPLE {i}: CHUNK {chunk.chunk_id} - Strategy: {chunk.primary_strategy.upper()}")
        print(f"{'─'*70}")
        
        print(f"\n Strategy Info:")
        print(f"  Primary Strategy:   {chunk.primary_strategy}")
        print(f"  All Strategies:     {', '.join(chunk.strategies_applied)}")
        
        print(f"\n Classification:")
        print(f"  Special Content:    {chunk.is_special_content}")
        print(f"  Semantic Unit:      {chunk.is_semantic_unit}")
        print(f"  Was Split:          {chunk.was_split}")
        print(f"  Has Overlap:        {chunk.has_overlap}")
        print(f"  Size Category:      {chunk.metadata.get('size_category', 'unknown')}")
        
        print(f"\nContext:")
        if chunk.heading_hierarchy:
            hierarchy_str = " > ".join(chunk.heading_hierarchy)
            print(f"  Hierarchy: {hierarchy_str}")
        print(f"  Section: {chunk.section_path}")
        print(f"  Pages: {chunk.page_numbers}")
        
        print(f"\nStats:")
        print(f"  Size: {len(chunk.content)} chars, {len(chunk.content.split())} words")
        print(f"  Content Types: {', '.join(chunk.content_types)}")
        print(f"  Blocks: {chunk.metadata.get('num_blocks', 0)}")
        
        print(f"\n Content Preview (first 250 chars):")
        preview = chunk.content[:250]
        if len(chunk.content) > 250:
            preview += "..."
        print(f"  {preview}")
        print()


def save_chunks(chunks: List[HybridChunk], output_path: Path):
    """Save chunks to JSONL file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
    
    print(f"\n Saved {len(chunks)} chunks to: {output_path}")


def save_embedding_format(chunks: List[HybridChunk], output_path: Path):
    """Save chunks in format optimized for embedding"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    embedding_data = []
    for chunk in chunks:
        embedding_data.append({
            'id': chunk.chunk_id,
            'text': chunk.full_text_with_context,
            'metadata': {
                'chunk_id': chunk.chunk_id,
                'strategy': chunk.primary_strategy,
                'section': chunk.section_path,
                'pages': chunk.page_numbers,
                'is_special': chunk.is_special_content,
                'char_count': len(chunk.content),
                'word_count': len(chunk.content.split())
            }
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in embedding_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 Saved embedding format to: {output_path}")


def main():
    """Main execution pipeline"""
    print("="*70)
    print("PHASE B: HYBRID INTELLIGENT CHUNKING")
    print("="*70)
    print()
    
    # Configuration
    input_path = Path("data/enhanced_pymupdf_output/enhanced_pymupdf_blocks.jsonl")
    output_dir = Path("data/chunking_output_pymudpdf")
    output_path = output_dir / "hybrid_intelligent_chunks.jsonl"
    embedding_path = output_dir / "hybrid_intelligent_chunks_for_embedding.jsonl"
    stats_path = output_dir / "chunk_statistics.csv"
    
    # Verify input exists
    if not input_path.exists():
        print(f" Error: Input file not found at {input_path}")
        print(f"   Please run Phase A parser first to generate the blocks file.")
        return
    
    # Run chunking
    print(f"📂 Input: {input_path}")
    print(f"📂 Output: {output_path}")
    print()
    
    hybrid_chunks = chunk_hybrid_intelligent(
        jsonl_path=str(input_path),
        target_size=1500,
        overlap=150,
        max_size=2500,
        min_size=300,
        keep_tables_whole=True,
        keep_code_whole=True,
        respect_sections=True,
        add_context=True,
        skip_types=['figure']
    )
    
    print(f"\nHybrid intelligent chunking complete!")
    print(f" Total chunks: {len(hybrid_chunks)}")
    
    # Analyze
    stats_df = analyze_hybrid_chunks(hybrid_chunks)
    
    # Preview
    preview_hybrid_chunks(hybrid_chunks, num_samples=5)
    
    # Save outputs
    save_chunks(hybrid_chunks, output_path)
    save_embedding_format(hybrid_chunks, embedding_path)
    
    # Save statistics
    stats_df.to_csv(stats_path, index=False)
    print(f" Saved statistics to: {stats_path}")
    
    print(f"\n{'='*70}")
    print(" PHASE B COMPLETE - READY FOR EMBEDDING")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()