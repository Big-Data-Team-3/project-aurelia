#!/usr/bin/env python3
"""
Enhanced PDF Parser Module - Cloud Composer Compatible
Full-featured parser with all sophisticated extraction capabilities
"""

import fitz
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from google.cloud import storage
from urllib.parse import urlparse


# Try to import optional dependencies
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False


class EnhancedPyMuPDFParser:
    """Enhanced PDF parser with full post-processing capabilities."""
    
    def __init__(self, pdf_path: Path, output_dir: Path, pdf_hash: str, 
                 config: Optional[Dict] = None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.pdf_hash = pdf_hash
        
        # Default configuration
        self.config = {
            'grouping_tolerance': 5,
            'min_group_size': 1,
            'font_similarity_threshold': 2,
            'min_heading_font_size': 12,
            'max_heading_word_count': 15,
            'heading_position_threshold': 150,
            'min_code_lines': 2,
            'code_keywords': [
                'function', 'def ', 'class ', 'import ', 'if __', 'for ', 'while ',
                'end', 'begin', 'return', 'var ', 'let ', 'const ', 'int ', 'float ',
                'string', 'array', 'object', 'void', 'public', 'private', 'static'
            ],
            'list_indicators': ['•', '-', '*', '◦', '▪', '1.', '2.', '3.', 'a.', 'b.', 'c.'],
            'min_list_items': 2,
            'use_camelot': True,
            'use_tabula': True,
            'table_confidence_threshold': 0.8,
            'min_figure_size': 20,
            'max_figure_page_ratio': 0.8,
            'exclude_edge_figures': True,
            'max_section_depth': 5,
            'simplify_section_paths': True
        }
        
        if config:
            self.config.update(config)
        
        self.blocks = []
        self.section_stack = []
        self.current_page = 0
        self.current_section = "root"
        self.stats = defaultdict(int)
        self.cleanup_stats = {
            'blocks_before_cleanup': 0,
            'blocks_after_cleanup': 0,
            'removed_blocks': 0,
            'removal_reasons': defaultdict(int)
        }
    
    def parse(self) -> List[Dict]:
        """Main parsing workflow with full enhancement pipeline"""
        print("🚀 Enhanced PyMuPDF Parser - Starting")
        
        doc = fitz.open(str(self.pdf_path))
        
        try:
            # Extract blocks from each page
            block_id = 0
            for page_num in range(len(doc)):
                self.current_page = page_num + 1
                page = doc[page_num]
                page_blocks = self._extract_page_blocks(page, page_num, block_id)
                self.blocks.extend(page_blocks)
                block_id += len(page_blocks)
            
            # Full enhancement pipeline
            print("🔧 Running enhancement pipeline...")
            self._post_process_blocks()
            self._augment_captions()
            self._validate_blocks()
            self._cleanup_blocks()
            
            print(f"✅ Parsed {len(self.blocks)} blocks")
            
        finally:
            doc.close()
        
        return self.blocks
    
    def _extract_page_blocks(self, page, page_num: int, start_block_id: int) -> List[Dict]:
        """Extract all blocks from a single page"""
        blocks = []
        block_id = start_block_id
        
        # Extract text blocks with enhancement
        text_blocks = self._extract_text_blocks_enhanced(page, page_num, block_id)
        blocks.extend(text_blocks)
        block_id += len(text_blocks)
        
        # Extract images/figures
        image_blocks = self._extract_image_blocks(page, page_num, block_id)
        blocks.extend(image_blocks)
        block_id += len(image_blocks)
        
        # Extract tables
        table_blocks = self._extract_table_blocks(page, page_num, block_id)
        blocks.extend(table_blocks)
        block_id += len(table_blocks)
        
        # Sort by vertical position
        blocks.sort(key=lambda b: b.get('bbox', {}).get('y0', 0), reverse=False)
        
        # Update block IDs after sorting
        for i, block in enumerate(blocks):
            block['block_id'] = start_block_id + i
        
        return blocks
    
    def _extract_text_blocks_enhanced(self, page, page_num: int, start_block_id: int) -> List[Dict]:
        """Enhanced text extraction with sophisticated grouping"""
        blocks = []
        block_id = start_block_id
        
        # Get text with formatting info
        text_dict = page.get_text("dict")
        
        # Extract all text elements with properties
        text_elements = []
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        text_elements.append({
                            "text": span["text"],
                            "bbox": span["bbox"],
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                            "color": span.get("color", 0)
                        })
        
        # Enhanced grouping
        grouped_blocks = self._enhanced_group_text_elements(text_elements)
        
        for group in grouped_blocks:
            block_text = group["text"].strip()
            if not block_text:
                continue
            
            bbox_dict = {
                "x0": round(group["bbox"][0], 2),
                "y0": round(group["bbox"][1], 2),
                "x1": round(group["bbox"][2], 2),
                "y1": round(group["bbox"][3], 2),
                "width": round(group["bbox"][2] - group["bbox"][0], 2),
                "height": round(group["bbox"][3] - group["bbox"][1], 2)
            }
            
            # Enhanced classification
            text_type = self._enhanced_classify_text_type(block_text, group)
            
            block_data = {
                "block_id": block_id,
                "type": text_type,
                "text": block_text,
                "page": page_num + 1,
                "bbox": bbox_dict,
                "char_count": len(block_text),
                "word_count": len(block_text.split()),
                "section_path": self._get_section_path(),
                "pdf_hash": self.pdf_hash,
                "level_in_doc": 1
            }
            
            # Add type-specific fields
            if text_type == "heading":
                block_data["heading_level"] = self._determine_heading_level(block_text, group)
                self._update_section_stack(block_text, block_data["heading_level"])
            elif text_type == "code":
                block_data["code_language"] = self._detect_code_language(block_text)
            elif text_type == "equation":
                block_data["latex"] = self._extract_latex(block_text)
            elif text_type == "list":
                block_data["list_type"] = self._detect_list_type(block_text)
            
            blocks.append(block_data)
            self.stats[text_type] += 1
            block_id += 1
        
        return blocks
    
    def _enhanced_group_text_elements(self, text_elements: List[Dict]) -> List[Dict]:
        """Enhanced text grouping with smart algorithms"""
        if not text_elements:
            return []
        
        # Sort by vertical then horizontal
        text_elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        
        grouped = []
        current_group = None
        
        for element in text_elements:
            if current_group is None:
                current_group = {
                    "text": element["text"],
                    "bbox": element["bbox"],
                    "font": element["font"],
                    "size": element["size"],
                    "flags": element["flags"],
                    "color": element["color"]
                }
            else:
                if self._should_group_elements_enhanced(current_group, element):
                    # Add to current group
                    current_group["text"] += " " + element["text"]
                    # Expand bounding box
                    current_group["bbox"] = [
                        min(current_group["bbox"][0], element["bbox"][0]),
                        min(current_group["bbox"][1], element["bbox"][1]),
                        max(current_group["bbox"][2], element["bbox"][2]),
                        max(current_group["bbox"][3], element["bbox"][3])
                    ]
                else:
                    # Finish current group and start new one
                    grouped.append(current_group)
                    current_group = {
                        "text": element["text"],
                        "bbox": element["bbox"],
                        "font": element["font"],
                        "size": element["size"],
                        "flags": element["flags"],
                        "color": element["color"]
                    }
        
        if current_group:
            grouped.append(current_group)
        
        return grouped
    
    def _should_group_elements_enhanced(self, current: Dict, element: Dict) -> bool:
        """Enhanced grouping logic with configurable parameters"""
        # Vertical proximity
        current_bottom = current["bbox"][3]
        element_top = element["bbox"][1]
        vertical_gap = element_top - current_bottom
        
        # Horizontal overlap
        current_left = current["bbox"][0]
        current_right = current["bbox"][2]
        element_left = element["bbox"][0]
        element_right = element["bbox"][2]
        
        horizontal_overlap = not (element_right < current_left or element_left > current_right)
        
        # Font similarity
        font_similar = (current["font"] == element["font"] and 
                       abs(current["size"] - element["size"]) < self.config['font_similarity_threshold'])
        
        # Enhanced grouping criteria
        tolerance = self.config['grouping_tolerance']
        
        return ((vertical_gap <= tolerance and horizontal_overlap) or
                (vertical_gap <= tolerance / 2) or
                (font_similar and vertical_gap <= tolerance * 2))
    
    def _enhanced_classify_text_type(self, text: str, group: Dict) -> str:
        """Enhanced text classification with improved rules"""
        # Check for code blocks first
        if self._is_code_block_enhanced(text, group):
            return "code"
        
        # Check for equations
        if self._is_equation_enhanced(text):
            return "equation"
        
        # Check for lists
        if self._is_list_enhanced(text):
            return "list"
        
        # Check for headings
        if self._is_heading_enhanced(text, group):
            return "heading"
        
        return "paragraph"
    
    def _is_code_block_enhanced(self, text: str, group: Dict) -> bool:
        """Enhanced code block detection"""
        text_lower = text.lower().strip()
        
        # Multi-line text with programming keywords
        if (text.count('\n') >= self.config['min_code_lines'] and 
            any(keyword in text_lower for keyword in self.config['code_keywords'])):
            return True
        
        # Monospace font
        font = group.get("font", "").lower()
        if any(mono in font for mono in ['courier', 'mono', 'fixed', 'code']):
            return True
        
        # Code patterns
        if re.search(r'[{}();]', text) and len(text.split()) > 3:
            return True
        
        return False
    
    def _is_equation_enhanced(self, text: str) -> bool:
        """Enhanced equation detection"""
        # Mathematical symbols
        math_symbols = ['=', '∑', '∫', '√', 'π', 'α', 'β', 'γ', 'δ', '±', '≤', '≥', '≠', '∞', '∂', '∇']
        if any(char in text for char in math_symbols):
            return True
        
        # Mathematical patterns
        if re.search(r'[a-zA-Z]\s*[=<>]\s*[a-zA-Z0-9]', text):
            return True
        
        # LaTeX patterns
        if re.search(r'\\[a-zA-Z]+', text):
            return True
        
        return False
    
    def _is_list_enhanced(self, text: str) -> bool:
        """Enhanced list detection"""
        text_stripped = text.strip()
        
        # Check for list indicators
        for indicator in self.config['list_indicators']:
            if text_stripped.startswith(indicator):
                return True
        
        # Check for numbered lists
        if re.match(r'^\s*\d+\.\s+', text_stripped):
            return True
        
        # Check for lettered lists
        if re.match(r'^\s*[a-zA-Z]\.\s+', text_stripped):
            return True
        
        # Check for multiple list items in one block
        lines = text.split('\n')
        list_items = 0
        for line in lines:
            line = line.strip()
            if any(line.startswith(indicator) for indicator in self.config['list_indicators']):
                list_items += 1
            elif re.match(r'^\s*\d+\.\s+', line):
                list_items += 1
        
        return list_items >= self.config['min_list_items']
    
    def _is_heading_enhanced(self, text: str, group: Dict) -> bool:
        """Enhanced heading detection"""
        text_lower = text.lower().strip()
        text_upper = text.upper().strip()
        word_count = len(text.split())
        font_size = group.get("size", 0)
        y_position = group["bbox"][1]
        
        # Exclude non-headings
        if self._is_non_heading_enhanced(text, group):
            return False
        
        # Strong indicators
        strong_indicators = [
            font_size > self.config['min_heading_font_size'] + 2,
            y_position < self.config['heading_position_threshold'],
            text_upper == text and 2 <= word_count <= 8,
            text_lower.startswith(('chapter', 'section', 'part', 'appendix')),
            bool(re.match(r'^\d+\.?\s+[A-Z]', text)),
            bool(re.match(r'^[A-Z][a-zA-Z\s]+:$', text))
        ]
        
        # Medium indicators
        medium_indicators = [
            font_size > self.config['min_heading_font_size'],
            text_upper == text and word_count <= 4,
            y_position < 400,
            bool(re.match(r'^[A-Z][a-z]+.*[A-Z]', text)) and word_count <= 6
        ]
        
        strong_count = sum(strong_indicators)
        medium_count = sum(medium_indicators)
        
        return strong_count >= 1 or medium_count >= 2
    
    def _is_non_heading_enhanced(self, text: str, group: Dict) -> bool:
        """Enhanced non-heading detection"""
        text_lower = text.lower().strip()
        word_count = len(text.split())
        
        # Version numbers
        if re.match(r'^[Rr]\s*\d{4}[a-z]?$', text):
            return True
        
        # URLs
        if 'www.' in text_lower or 'http' in text_lower:
            return True
        
        # Very long text
        if word_count > self.config['max_heading_word_count']:
            return True
        
        # Email addresses
        if '@' in text:
            return True
        
        # Page numbers
        if text.isdigit() and len(text) <= 3:
            return True
        
        return False
    
    def _extract_image_blocks(self, page, page_num: int, start_block_id: int) -> List[Dict]:
        """Extract images with filtering"""
        blocks = []
        block_id = start_block_id
        
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            img_rects = page.get_image_rects(img)
            if not img_rects:
                continue
                
            for rect in img_rects:
                rect_width = rect.width
                rect_height = rect.height
                
                # Filtering
                if (rect_width > page_width * self.config['max_figure_page_ratio'] and 
                    rect_height > page_height * self.config['max_figure_page_ratio']):
                    continue
                
                if (rect_width < self.config['min_figure_size'] or 
                    rect_height < self.config['min_figure_size']):
                    continue
                
                if self.config['exclude_edge_figures']:
                    if (rect.x0 < 10 or rect.y0 < 10 or 
                        rect.x1 > page_width - 10 or rect.y1 > page_height - 10):
                        continue
                
                bbox_dict = {
                    "x0": round(rect.x0, 2),
                    "y0": round(rect.y0, 2),
                    "x1": round(rect.x1, 2),
                    "y1": round(rect.y1, 2),
                    "width": round(rect_width, 2),
                    "height": round(rect_height, 2)
                }
                
                block_data = {
                    "block_id": block_id,
                    "type": "figure",
                    "text": "[Figure]",
                    "page": page_num + 1,
                    "bbox": bbox_dict,
                    "caption": None,
                    "figure_id": f"#/pictures/{img_index}",
                    "has_caption": False,
                    "section_path": self._get_section_path(),
                    "pdf_hash": self.pdf_hash,
                    "level_in_doc": 1
                }
                
                blocks.append(block_data)
                self.stats["figure"] += 1
                block_id += 1
        
        return blocks
    
    def _extract_table_blocks(self, page, page_num: int, start_block_id: int) -> List[Dict]:
        """Extract tables"""
        blocks = []
        block_id = start_block_id
        
        try:
            tables = page.find_tables()
            
            for table_index, table in enumerate(tables):
                table_data = table.extract()
                if not table_data:
                    continue
                
                markdown = self._table_to_markdown(table_data)
                bbox = table.bbox
                bbox_dict = {
                    "x0": round(bbox[0], 2),
                    "y0": round(bbox[1], 2),
                    "x1": round(bbox[2], 2),
                    "y1": round(bbox[3], 2),
                    "width": round(bbox[2] - bbox[0], 2),
                    "height": round(bbox[3] - bbox[1], 2)
                }
                
                block_data = {
                    "block_id": block_id,
                    "type": "table",
                    "text": markdown,
                    "page": page_num + 1,
                    "bbox": bbox_dict,
                    "caption": None,
                    "table_id": f"#/tables/{table_index}",
                    "has_caption": False,
                    "num_rows": len(table_data),
                    "num_cols": len(table_data[0]) if table_data else 0,
                    "section_path": self._get_section_path(),
                    "pdf_hash": self.pdf_hash,
                    "level_in_doc": 1
                }
                
                blocks.append(block_data)
                self.stats["table"] += 1
                block_id += 1
                
        except Exception as e:
            print(f"⚠️ Error extracting tables from page {page_num + 1}: {e}")
        
        return blocks
    
    def _post_process_blocks(self):
        """Post-processing pipeline"""
        # Merge related tables
        self._merge_related_tables()
        
        # Enhance list detection
        self._enhance_list_detection()
        
        # Refine code blocks
        self._refine_code_blocks()
        
        # Simplify section paths
        if self.config['simplify_section_paths']:
            self._simplify_section_paths()
    
    def _merge_related_tables(self):
        """Merge tables with same headers"""
        merged_tables = 0
        i = 0
        
        while i < len(self.blocks):
            block = self.blocks[i]
            
            if block.get('type') != 'table':
                i += 1
                continue
            
            j = i + 1
            while j < len(self.blocks):
                next_block = self.blocks[j]
                
                if next_block.get('type') != 'table':
                    j += 1
                    continue
                
                if self._can_merge_tables(block, next_block):
                    merged_table = self._merge_two_tables(block, next_block)
                    self.blocks[i] = merged_table
                    self.blocks.pop(j)
                    merged_tables += 1
                    break
                else:
                    j += 1
            
            i += 1
        
        if merged_tables > 0:
            print(f"  Merged {merged_tables} related tables")
    
    def _can_merge_tables(self, table1: Dict, table2: Dict) -> bool:
        """Check if two tables can be merged"""
        page1 = table1.get('page', 0)
        page2 = table2.get('page', 0)
        
        if abs(page1 - page2) > 1:
            return False
        
        cols1 = table1.get('num_cols', 0)
        cols2 = table2.get('num_cols', 0)
        
        if cols1 != cols2 or cols1 == 0:
            return False
        
        text1 = table1.get('text', '')
        text2 = table2.get('text', '')
        
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        if len(lines1) < 2 or len(lines2) < 2:
            return False
        
        header1 = lines1[0]
        header2 = lines2[0]
        
        return self._headers_similar(header1, header2)
    
    def _headers_similar(self, header1: str, header2: str) -> bool:
        """Check if headers are similar"""
        h1 = header1.lower().strip()
        h2 = header2.lower().strip()
        
        if h1 == h2:
            return True
        
        # Check common patterns
        if 'dateformnum' in h1 and 'dateformnum' in h2:
            if 'example' in h1 and 'example' in h2:
                return True
        
        return False
    
    def _merge_two_tables(self, table1: Dict, table2: Dict) -> Dict:
        """Merge two tables"""
        text1 = table1.get('text', '')
        text2 = table2.get('text', '')
        
        lines1 = text1.split('\n')
        lines2 = text2.split('\n')
        
        data_lines2 = lines2[2:] if len(lines2) > 2 else []
        merged_lines = lines1 + data_lines2
        
        merged_table = table1.copy()
        merged_table['text'] = '\n'.join(merged_lines)
        merged_table['num_rows'] = len(merged_lines) - 1
        
        page1 = table1.get('page', 0)
        page2 = table2.get('page', 0)
        if page1 != page2:
            merged_table['page_range'] = f"{min(page1, page2)}-{max(page1, page2)}"
        
        return merged_table
    
    def _enhance_list_detection(self):
        """Enhance list detection"""
        enhanced_lists = 0
        
        for block in self.blocks:
            if block['type'] == 'paragraph':
                text = block['text']
                lines = text.split('\n')
                list_items = []
                
                for line in lines:
                    line = line.strip()
                    if any(line.startswith(indicator) for indicator in self.config['list_indicators']):
                        list_items.append(line)
                    elif re.match(r'^\s*\d+\.\s+', line):
                        list_items.append(line)
                
                if len(list_items) >= self.config['min_list_items']:
                    block['type'] = 'list'
                    block['list_type'] = self._detect_list_type(text)
                    block['list_items'] = list_items
                    enhanced_lists += 1
                    self.stats['list'] += 1
                    self.stats['paragraph'] -= 1
        
        if enhanced_lists > 0:
            print(f"  Enhanced {enhanced_lists} lists")
    
    def _refine_code_blocks(self):
        """Refine code block classification"""
        refined_blocks = 0
        
        for block in self.blocks:
            if block['type'] == 'code':
                text = block['text']
                
                if self._is_false_positive_code(text):
                    block['type'] = 'paragraph'
                    refined_blocks += 1
                    self.stats['code'] -= 1
                    self.stats['paragraph'] += 1
        
        if refined_blocks > 0:
            print(f"  Refined {refined_blocks} false positive code blocks")
    
    def _is_false_positive_code(self, text: str) -> bool:
        """Check if code block is false positive"""
        text_lower = text.lower().strip()
        
        if ('www.' in text_lower or 'http' in text_lower) and len(text.split()) <= 3:
            return True
        
        if len(text.split()) == 1 and not any(char in text for char in ['{}();']):
            return True
        
        if len(text) < 10 and not re.search(r'[{}();]', text):
            return True
        
        return False
    
    def _simplify_section_paths(self):
        """Simplify section paths"""
        for block in self.blocks:
            if 'section_path' in block:
                path = block['section_path']
                
                if self.config['simplify_section_paths']:
                    path_parts = path.split(' > ')
                    if len(path_parts) > self.config['max_section_depth']:
                        path_parts = path_parts[:self.config['max_section_depth']]
                        block['section_path'] = ' > '.join(path_parts)
    
    def _augment_captions(self):
        """Augment captions for figures and tables"""
        for i, block in enumerate(self.blocks):
            if block['type'] not in ['figure', 'table']:
                continue
            if block.get('caption'):
                continue
            
            caption = None
            
            # Look backward for heading
            for j in range(i-1, max(i-30, -1), -1):
                prev = self.blocks[j]
                if prev['page'] != block['page']:
                    break
                if prev['type'] == 'heading' and prev.get('heading_level', 99) <= 3:
                    caption = prev['text']
                    break
            
            # Look forward for caption text
            if not caption:
                for j in range(i+1, min(i+5, len(self.blocks))):
                    next_block = self.blocks[j]
                    if next_block['page'] != block['page']:
                        break
                    text = next_block.get('text', '').strip()
                    if text.startswith(('Figure', 'Table', 'Fig.')):
                        caption = text.split('\n')[0]
                        break
            
            # Use section path as fallback
            if not caption:
                caption = block['section_path']
            
            block['caption'] = caption
            block['caption_source'] = 'augmented'
            block['has_caption'] = True
    
    def _validate_blocks(self):
        """Validate blocks"""
        valid_blocks = []
        for block in self.blocks:
            if block['type'] in ['paragraph', 'heading'] and not block.get('text', '').strip():
                continue
            
            if 'block_id' not in block or 'type' not in block:
                continue
            
            valid_blocks.append(block)
        
        self.blocks = valid_blocks
    
    def _cleanup_blocks(self):
        """Cleanup blocks"""
        self.cleanup_stats['blocks_before_cleanup'] = len(self.blocks)
        
        cleaned_blocks = []
        for block in self.blocks:
            should_keep, reason = self._should_keep_block(block)
            
            if should_keep:
                cleaned_block = self._clean_block_metadata(block)
                cleaned_blocks.append(cleaned_block)
            else:
                self.cleanup_stats['removal_reasons'][reason] += 1
        
        self.blocks = cleaned_blocks
        self.cleanup_stats['blocks_after_cleanup'] = len(self.blocks)
        self.cleanup_stats['removed_blocks'] = (
            self.cleanup_stats['blocks_before_cleanup'] - 
            self.cleanup_stats['blocks_after_cleanup']
        )
    
    def _should_keep_block(self, block: Dict) -> Tuple[bool, str]:
        """Determine if block should be kept"""
        block_type = block.get('type', 'unknown')
        text = block.get('text', '').strip()
        char_count = block.get('char_count', 0)
        
        if text == '.' and block_type == 'paragraph':
            return False, 'single_dot_paragraph'
        
        if block_type == 'paragraph' and char_count < 3:
            if not text.isdigit() and text not in ['-', '—', '–', '*', '•']:
                return False, 'too_small_meaningless'
        
        if text in ['', ' ', '\n', '\t']:
            return False, 'no_meaningful_content'
        
        if text.count('.') > 10 and len(text.replace('.', '').strip()) < 5:
            return False, 'excessive_dots'
        
        return True, ''
    
    def _clean_block_metadata(self, block: Dict) -> Dict:
        """Clean block metadata"""
        cleaned = block.copy()
        
        if 'bbox' in cleaned and cleaned['bbox']:
            bbox = cleaned['bbox']
            for key in ['x0', 'y0', 'x1', 'y1', 'width', 'height']:
                if key in bbox and isinstance(bbox[key], (int, float)):
                    bbox[key] = round(bbox[key], 2)
            cleaned['bbox'] = bbox
        
        return cleaned
    
    def _determine_heading_level(self, text: str, group: Dict) -> int:
        """Determine heading level"""
        text_lower = text.lower().strip()
        font_size = group.get("size", 0)
        word_count = len(text.split())
        
        # Level 1: Major sections
        if (font_size > 16 or
            text_lower.startswith(('chapter', 'part', 'appendix')) or
            re.match(r'^\d+\.?\s+[A-Z]', text) or
            (word_count <= 3 and text.isupper())):
            return 1
        
        # Level 2: Sub-sections
        elif (font_size > 14 or
              text_lower.startswith('section') or
              re.match(r'^\d+\.\d+', text) or
              (word_count <= 5 and font_size > 12)):
            return 2
        
        # Level 3: Sub-sub-sections
        elif (font_size > 12 or
              re.match(r'^\d+\.\d+\.\d+', text) or
              (word_count <= 8 and font_size > 10)):
            return 3
        
        # Level 4: Minor headings
        elif font_size > 10 or word_count <= 10:
            return 4
        
        else:
            return 1
    
    def _get_section_path(self) -> str:
        """Get current section path"""
        return self.current_section
    
    def _update_section_stack(self, text: str, level: int):
        """Update section hierarchy"""
        if not text or len(text) < 3:
            return
        
        clean_text = text.strip().replace('\n', ' ').strip()
        if len(clean_text) > 100:
            clean_text = clean_text[:100]
        
        if clean_text and len(clean_text) > 2:
            self.current_section = clean_text
    
    def _detect_code_language(self, text: str) -> str:
        """Detect programming language"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['def ', 'import ', 'from ']):
            return 'python'
        elif any(keyword in text_lower for keyword in ['function', 'var ', 'let ', 'const ']):
            return 'javascript'
        elif any(keyword in text_lower for keyword in ['class ', 'public', 'private', 'static']):
            return 'java'
        elif any(keyword in text_lower for keyword in ['#include', 'int ', 'float ', 'void ']):
            return 'c'
        else:
            return 'unknown'
    
    def _extract_latex(self, text: str) -> Optional[str]:
        """Extract LaTeX from equation"""
        latex_match = re.search(r'\\[a-zA-Z]+.*', text)
        if latex_match:
            return latex_match.group(0)
        return None
    
    def _detect_list_type(self, text: str) -> str:
        """Detect list type"""
        text_stripped = text.strip()
        
        if text_stripped.startswith(('•', '-', '*', '◦', '▪')):
            return 'bullet'
        elif re.match(r'^\s*\d+\.\s+', text_stripped):
            return 'numbered'
        elif re.match(r'^\s*[a-zA-Z]\.\s+', text_stripped):
            return 'lettered'
        else:
            return 'mixed'
    
    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert table to markdown"""
        if not table_data:
            return ""
        
        cleaned_data = []
        for row in table_data:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_data.append(cleaned_row)
        
        if not cleaned_data:
            return ""
        
        markdown_lines = []
        
        # Header
        header = "| " + " | ".join(cleaned_data[0]) + " |"
        markdown_lines.append(header)
        
        # Separator
        separator = "| " + " | ".join(["---"] * len(cleaned_data[0])) + " |"
        markdown_lines.append(separator)
        
        # Data rows
        for row in cleaned_data[1:]:
            if row:
                data_row = "| " + " | ".join(row) + " |"
                markdown_lines.append(data_row)
        
        return "\n".join(markdown_lines)


# ============================================================================
# CLOUD COMPOSER WRAPPER FUNCTIONS
# ============================================================================

def parse_pdf_from_gcs(pdf_gs_url: str, output_gs_url: str, config: Optional[Dict] = None) -> Dict:
    """
    Cloud Composer entry point: Parse PDF from GCS with full enhancement pipeline.
    
    Args:
        pdf_gs_url: Input PDF location (gs://bucket/path.pdf)
        output_gs_url: Output blocks location (gs://bucket/output.jsonl)
        config: Optional parser config
    
    Returns:
        Statistics dict
    """
    # Download PDF
    u = urlparse(pdf_gs_url)
    tmp_pdf = f"/tmp/{os.path.basename(u.path)}"
    
    storage.Client().bucket(u.netloc).blob(u.path.lstrip("/")).download_to_filename(tmp_pdf)
    
    # Parse with full enhancement
    tmp_output = Path("/tmp/parser_out")
    tmp_output.mkdir(exist_ok=True)
    
    parser = EnhancedPyMuPDFParser(
        pdf_path=Path(tmp_pdf),
        output_dir=tmp_output,
        pdf_hash=os.path.basename(u.path).replace('.pdf', ''),
        config=config
    )
    
    blocks = parser.parse()
    
    # Write blocks
    tmp_blocks = "/tmp/blocks.jsonl"
    with open(tmp_blocks, 'w', encoding='utf-8') as f:
        for block in blocks:
            f.write(json.dumps(block, ensure_ascii=False) + '\n')
    
    # Upload to GCS
    u_out = urlparse(output_gs_url)
    storage.Client().bucket(u_out.netloc).blob(u_out.path.lstrip("/")).upload_from_filename(tmp_blocks)
    
    # Cleanup
    os.remove(tmp_pdf)
    os.remove(tmp_blocks)
    
    return {
        'source': pdf_gs_url,
        'output': output_gs_url,
        'blocks_count': len(blocks),
        'stats': dict(parser.stats),
        'cleanup': parser.cleanup_stats
    }