#!/usr/bin/env python3
"""
Phase A Parser - Enhanced PDF parsing with Docling
Extracts blocks from PDF and saves to JSONL format
"""

from docling.document_converter import DocumentConverter
from docling_core.types.doc import TextItem, TableItem, PictureItem
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class PhaseAParser:
    """Enhanced parser for Phase A with better error handling and features."""
    
    def __init__(self, pdf_path: Path, output_dir: Path, pdf_hash: str, strict_mode: bool = False):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.pdf_hash = pdf_hash
        self.strict_mode = strict_mode
        self.blocks = []
        self.section_stack = []
        self.stats = defaultdict(int)
        self.cleanup_stats = {
            'blocks_before_cleanup': 0,
            'blocks_after_cleanup': 0,
            'removed_blocks': 0,
            'removal_reasons': defaultdict(int)
        }
        
    def parse(self) -> List[Dict]:
        """Main parsing workflow."""
        print("="*60)
        print("PHASE A: PARSE & NORMALIZE (ENHANCED)")
        print("="*60)
        
        # Convert
        converter = DocumentConverter()
        result = converter.convert(str(self.pdf_path))
        doc = result.document
        
        # Extract blocks
        block_id = 0
        for item, level in doc.iterate_items():
            try:
                block = self._extract_block(item, level, block_id, doc)
                if block:
                    self.blocks.append(block)
                    self.stats[block['type']] += 1
                    block_id += 1
            except Exception as e:
                print(f"⚠️  Error processing block {block_id}: {e}")
                continue
        
        # Post-processing
        self._augment_captions()
        self._validate_blocks()
        self._cleanup_blocks()
        
        # Save
        self._save_output()
        self._save_cleanup_report()
        self._print_stats()
        
        return self.blocks
    
    def _extract_block(self, item, level: int, block_id: int, doc) -> Optional[Dict]:
        """Extract single block with full metadata."""
        
        # Get provenance metadata
        page, bbox = self._extract_provenance(item)
        
        # Type and text extraction
        if isinstance(item, PictureItem):
            block = self._extract_figure(item, doc, block_id, page, bbox)
        elif isinstance(item, TableItem):
            block = self._extract_table(item, doc, block_id, page, bbox)
        elif isinstance(item, TextItem):
            block = self._extract_text(item, doc, block_id, page, bbox, level)
        else:
            return None
        
        # Add common metadata
        block.update({
            "section_path": " > ".join(self.section_stack) if self.section_stack else "root",
            "pdf_hash": self.pdf_hash,
            "level_in_doc": level  # Original document level
        })
        
        return block
    
    def _extract_provenance(self, item) -> Tuple[Optional[int], Optional[Dict]]:
        """Extract page and bounding box from provenance."""
        page, bbox = None, None
        if hasattr(item, 'prov') and item.prov:
            prov = item.prov[0]
            page = getattr(prov, 'page_no', None)
            if hasattr(prov, 'bbox'):
                b = prov.bbox
                bbox = {
                    "x0": float(b.l), 
                    "y0": float(b.t), 
                    "x1": float(b.r), 
                    "y1": float(b.b),
                    "width": float(b.r - b.l),
                    "height": float(b.t - b.b)
                }
        return page, bbox
    
    def _extract_figure(self, item: PictureItem, doc, block_id: int, 
                       page: int, bbox: Dict) -> Dict:
        """Extract figure block."""
        return {
            "block_id": block_id,
            "type": "figure",
            "text": "[Figure]",
            "page": page,
            "bbox": bbox,
            "caption": item.caption_text(doc=doc),
            "figure_id": str(item.self_ref),
            "has_caption": bool(item.caption_text(doc=doc))
        }
    
    def _extract_table(self, item: TableItem, doc, block_id: int,
                      page: int, bbox: Dict) -> Dict:
        """Extract table block."""
        markdown = item.export_to_markdown(doc=doc)
        return {
            "block_id": block_id,
            "type": "table",
            "text": markdown,
            "page": page,
            "bbox": bbox,
            "caption": item.caption_text(doc=doc),
            "table_id": str(item.self_ref),
            "has_caption": bool(item.caption_text(doc=doc)),
            "num_rows": markdown.count('\n') if markdown else 0,
            "num_cols": len(markdown.split('\n')[0].split('|')) - 2 if markdown and '\n' in markdown else 0
        }
    
    def _extract_text(self, item: TextItem, doc, block_id: int,
                     page: int, bbox: Dict, level: int) -> Dict:
        """Extract text-based block (heading, paragraph, code, etc)."""
        
        # Classify type
        label = str(item.label).lower() if hasattr(item, 'label') else ''
        text = item.text or ""
        
        if 'section' in label or 'title' in label:
            item_type = 'heading'
            self._update_section_stack(text, level)
        elif 'code' in label:
            item_type = 'code'
        elif 'formula' in label or 'equation' in label:
            item_type = 'equation'
        elif 'list' in label or text.strip().startswith(('•', '-', '*', '1.', '2.')):
            item_type = 'list'
        else:
            item_type = 'paragraph'
        
        block = {
            "block_id": block_id,
            "type": item_type,
            "text": text,
            "page": page,
            "bbox": bbox,
            "char_count": len(text),
            "word_count": len(text.split())
        }
        
        # Type-specific fields
        if item_type == "heading":
            block["heading_level"] = level
        elif item_type == "code":
            block["code_language"] = getattr(item, 'code_language', 'unknown')
        elif item_type == "equation":
            block["latex"] = getattr(item, 'latex', None)
        
        return block
    
    def _update_section_stack(self, text: str, level: int):
        """Update section hierarchy stack."""
        if not text or len(text) < 3:
            return
        
        # Adjust stack based on level
        if level <= len(self.section_stack):
            self.section_stack = self.section_stack[:level-1]
        
        self.section_stack.append(text[:100])  # Truncate long headings
    
    def _augment_captions(self):
        """Enhanced caption augmentation with multiple strategies."""
        
        for i, block in enumerate(self.blocks):
            if block['type'] not in ['figure', 'table']:
                continue
            if block.get('caption'):
                continue
            
            caption = None
            
            # Strategy 1: Look backward for heading on same page
            for j in range(i-1, max(i-30, -1), -1):
                prev = self.blocks[j]
                if prev['page'] != block['page']:
                    break
                if prev['type'] == 'heading' and prev.get('heading_level', 99) <= 3:
                    caption = prev['text']
                    break
            
            # Strategy 2: Look forward for paragraph starting with "Figure" or "Table"
            if not caption:
                for j in range(i+1, min(i+5, len(self.blocks))):
                    next_block = self.blocks[j]
                    if next_block['page'] != block['page']:
                        break
                    text = next_block.get('text', '').strip()
                    if text.startswith(('Figure', 'Table', 'Fig.')):
                        caption = text.split('\n')[0]  # First line only
                        break
            
            # Strategy 3: Use section path as fallback
            if not caption:
                caption = block['section_path']
            
            block['caption'] = caption
            block['caption_source'] = 'augmented'
            block['has_caption'] = True
    
    def _validate_blocks(self):
        """Validate and clean blocks."""
        valid_blocks = []
        for block in self.blocks:
            # Skip empty text blocks
            if block['type'] in ['paragraph', 'heading'] and not block.get('text', '').strip():
                continue
            
            # Ensure required fields
            if 'block_id' not in block or 'type' not in block:
                continue
            
            valid_blocks.append(block)
        
        self.blocks = valid_blocks
    
    def _should_keep_block(self, block: Dict) -> Tuple[bool, str]:
        """
        Determine if a block should be kept or removed.
        
        Returns:
            Tuple of (should_keep, reason_if_removed)
        """
        block_type = block.get('type', 'unknown')
        text = block.get('text', '').strip()
        char_count = block.get('char_count', 0)
        word_count = block.get('word_count', 0)
        
        # RULE 1: Remove single-dot paragraphs
        if text == '.' and block_type == 'paragraph':
            return False, 'single_dot_paragraph'
        
        # RULE 2: Remove very small meaningless text (< 3 chars, excluding numbers)
        if block_type == 'paragraph' and char_count < 3:
            # Keep if it's a number or meaningful
            if not text.isdigit() and text not in ['-', '–', '—', '*', '•']:
                return False, 'too_small_meaningless'
        
        # RULE 3: Remove standalone page number paragraphs (optional in strict mode)
        if self.strict_mode and block_type == 'paragraph' and text.isdigit() and len(text) <= 2:
            return False, 'standalone_page_number'
        
        # RULE 4: Remove figures with only augmented captions (optional in strict mode)
        if self.strict_mode and block_type == 'figure':
            if block.get('caption_source') == 'augmented' and block.get('text') == '[Figure]':
                return False, 'figure_augmented_caption_only'
        
        # RULE 5: Remove blocks with no meaningful content
        if text in ['', ' ', '\n', '\t']:
            return False, 'no_meaningful_content'
        
        # RULE 6: Remove excessive ellipsis/dots patterns
        if text.count('.') > 10 and len(text.replace('.', '').strip()) < 5:
            return False, 'excessive_dots'
        
        return True, ''
    
    def _clean_block_metadata(self, block: Dict) -> Dict:
        """Clean and optimize block metadata."""
        cleaned = block.copy()
        
        # Round bbox coordinates to 2 decimal places (reduce precision bloat)
        if 'bbox' in cleaned and cleaned['bbox']:
            bbox = cleaned['bbox']
            for key in ['x0', 'y0', 'x1', 'y1', 'width', 'height']:
                if key in bbox and isinstance(bbox[key], (int, float)):
                    bbox[key] = round(bbox[key], 2)
            cleaned['bbox'] = bbox
        
        # Remove sample_hash if it's just a placeholder
        if cleaned.get('pdf_hash') == 'sample_hash':
            cleaned.pop('pdf_hash', None)
        
        return cleaned
    
    def _cleanup_blocks(self):
        """Apply cleanup rules to all blocks."""
        self.cleanup_stats['blocks_before_cleanup'] = len(self.blocks)
        
        cleaned_blocks = []
        for block in self.blocks:
            # Check if block should be kept
            should_keep, reason = self._should_keep_block(block)
            
            if should_keep:
                # Clean metadata
                cleaned_block = self._clean_block_metadata(block)
                cleaned_blocks.append(cleaned_block)
            else:
                # Track removal reason
                self.cleanup_stats['removal_reasons'][reason] += 1
        
        self.blocks = cleaned_blocks
        self.cleanup_stats['blocks_after_cleanup'] = len(self.blocks)
        self.cleanup_stats['removed_blocks'] = (
            self.cleanup_stats['blocks_before_cleanup'] - 
            self.cleanup_stats['blocks_after_cleanup']
        )
    
    def _save_output(self):
        """Save blocks to JSONL."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / "docling_blocks.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for block in self.blocks:
                f.write(json.dumps(block, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Saved: {output_file}")
    
    def _save_cleanup_report(self):
        """Save cleanup report with statistics."""
        report_file = self.output_dir / "cleanup_report.json"
        
        report = {
            'input_file': str(self.pdf_path),
            'output_file': str(self.output_dir / "docling_blocks.jsonl"),
            'strict_mode': self.strict_mode,
            'statistics': dict(self.stats),
            'cleanup': {
                'blocks_before': self.cleanup_stats['blocks_before_cleanup'],
                'blocks_after': self.cleanup_stats['blocks_after_cleanup'],
                'removed': self.cleanup_stats['removed_blocks'],
                'removal_reasons': dict(self.cleanup_stats['removal_reasons'])
            },
            'rules_applied': [
                "Remove single-dot paragraphs",
                "Remove very small meaningless text (<3 chars)",
                "Remove excessive ellipsis patterns",
                "Round bbox coordinates to 2 decimals",
                "Remove placeholder pdf_hash"
            ]
        }
        
        if self.strict_mode:
            report['rules_applied'].extend([
                "Remove standalone page numbers",
                "Remove figures with only augmented captions"
            ])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Report: {report_file}")
    
    def _print_stats(self):
        """Print comprehensive statistics."""
        fig_tab = [b for b in self.blocks if b['type'] in ['figure', 'table']]
        with_caps = [b for b in fig_tab if b.get('has_caption')]
        
        print(f"\n{'='*60}")
        print("EXTRACTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total blocks: {len(self.blocks)}")
        print(f"\nBlock types:")
        for block_type, count in sorted(self.stats.items()):
            print(f"  {block_type:12s}: {count:4d}")
        print(f"\nFigures/Tables: {len(fig_tab)}")
        print(f"With captions:  {len(with_caps)} ({len(with_caps)/len(fig_tab)*100:.1f}%)" if fig_tab else "With captions:  0")
        print(f"\nPages covered: {len(set(b['page'] for b in self.blocks if b.get('page')))}")
        
        # Cleanup statistics
        print(f"\n{'='*60}")
        print("CLEANUP STATISTICS")
        print(f"{'='*60}")
        print(f"Blocks before cleanup:  {self.cleanup_stats['blocks_before_cleanup']}")
        print(f"Blocks after cleanup:   {self.cleanup_stats['blocks_after_cleanup']}")
        print(f"Blocks removed:         {self.cleanup_stats['removed_blocks']}")
        
        if self.cleanup_stats['removal_reasons']:
            print(f"\nRemoval reasons:")
            for reason, count in sorted(self.cleanup_stats['removal_reasons'].items()):
                print(f"  {reason:30s}: {count:4d}")
        
        print(f"\n{'='*60}")
        print("READY FOR PHASE B: CHUNKING EXPERIMENTS")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    # Configuration
    pdf_path = Path("data/fintbx_ex.pdf")
    output_dir = Path("data/final")
    pdf_hash = "sample_hash"  # Replace with actual hash computation if needed
    strict_mode = False  # Set to True for more aggressive cleanup
    
    # Verify PDF exists
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found at {pdf_path}")
        print(f"   Please update the pdf_path variable with the correct path.")
        return
    
    # Run parser
    parser = PhaseAParser(
        pdf_path=pdf_path,
        output_dir=output_dir,
        pdf_hash=pdf_hash,
        strict_mode=strict_mode
    )
    
    blocks = parser.parse()
    
    print(f"\n✅ Successfully parsed {len(blocks)} blocks")
    print(f"   Output saved to: {output_dir / 'docling_blocks.jsonl'}")
    print(f"   Report saved to: {output_dir / 'cleanup_report.json'}")


if __name__ == "__main__":
    main()