"""
Wikipedia Service for Knowledge Fallback
Provides Wikipedia search as a fallback knowledge source
"""

import time
import asyncio
import re
from typing import List, Tuple, Optional
import wikipedia
import logging
from urllib.parse import quote

from config.rag_config import rag_config
from models.rag_models import SearchResult, SourceType


logger = logging.getLogger(__name__)


class WikipediaService:
    """Service for Wikipedia search as knowledge fallback"""
    
    def __init__(self):
        # Configure wikipedia library
        wikipedia.set_lang("en")
        wikipedia.set_rate_limiting(True)
        self.max_summary_length = 1000
        self.max_content_length = 2000
    
    async def search_wikipedia(
        self,
        query: str,
        top_k: int = 3,
        auto_suggest: bool = True
    ) -> Tuple[List[SearchResult], float]:
        """
        Search Wikipedia for relevant articles
        
        Args:
            query: Search query
            top_k: Number of results to return
            auto_suggest: Whether to use Wikipedia's auto-suggestion
            
        Returns:
            Tuple of (search results, search time in ms)
        """
        start_time = time.time()
        
        if not rag_config.enable_wikipedia_fallback:
            logger.info("Wikipedia fallback is disabled")
            return [], 0.0
        
        try:
            # Clean and prepare query
            clean_query = self._clean_query(query)
            
            # Search for Wikipedia pages
            search_results = wikipedia.search(clean_query, results=top_k * 2)
            
            if not search_results:
                logger.info(f"No Wikipedia results found for query: {query}")
                return [], (time.time() - start_time) * 1000
            
            # Get detailed information for each result
            results = []
            for title in search_results[:top_k]:
                try:
                    result = await self._get_wikipedia_page_info(title, query)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to get Wikipedia page info for '{title}': {e}")
                    continue
            
            search_time = (time.time() - start_time) * 1000
            logger.info(f"Wikipedia search completed in {search_time:.2f}ms, found {len(results)} results")
            
            return results, search_time
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return [], (time.time() - start_time) * 1000
    
    async def _get_wikipedia_page_info(
        self,
        title: str,
        original_query: str
    ) -> Optional[SearchResult]:
        """
        Get detailed information for a Wikipedia page
        
        Args:
            title: Wikipedia page title
            original_query: Original search query for relevance scoring
            
        Returns:
            SearchResult object or None if failed
        """
        try:
            # Get page object
            page = wikipedia.page(title)
            
            # Get summary (limited length)
            summary = wikipedia.summary(title, sentences=3)
            if len(summary) > self.max_summary_length:
                summary = summary[:self.max_summary_length] + "..."
            
            # Get relevant content sections
            content = self._extract_relevant_content(page, original_query)
            
            # Combine summary and relevant content
            full_text = f"{summary}\n\n{content}" if content else summary
            
            # Limit total text length
            if len(full_text) > self.max_content_length:
                full_text = full_text[:self.max_content_length] + "..."
            
            # Calculate relevance score based on query match
            relevance_score = self._calculate_relevance_score(full_text, original_query)
            
            # Create SearchResult
            result = SearchResult(
                id=f"wikipedia_{quote(title)}",
                text=full_text,
                score=relevance_score,
                source_type=SourceType.WIKIPEDIA,
                metadata={
                    'categories': getattr(page, 'categories', [])[:5],  # Limit categories
                    'links': len(getattr(page, 'links', [])),
                    'references': len(getattr(page, 'references', [])),
                    'summary': summary
                },
                title=page.title,
                url=page.url
            )
            
            return result
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try the first disambiguation option
            try:
                if e.options:
                    return await self._get_wikipedia_page_info(e.options[0], original_query)
            except Exception:
                pass
            logger.warning(f"Disambiguation error for '{title}': {e}")
            return None
            
        except wikipedia.exceptions.PageError:
            logger.warning(f"Wikipedia page not found: '{title}'")
            return None
            
        except Exception as e:
            logger.error(f"Error getting Wikipedia page info for '{title}': {e}")
            return None
    
    def _clean_query(self, query: str) -> str:
        """Clean and prepare query for Wikipedia search"""
        # Remove special characters and extra whitespace
        clean_query = re.sub(r'[^\w\s]', ' ', query)
        clean_query = ' '.join(clean_query.split())
        
        # Limit query length
        if len(clean_query) > 100:
            clean_query = clean_query[:100]
        
        return clean_query
    
    def _extract_relevant_content(self, page, query: str) -> str:
        """Extract relevant content sections from Wikipedia page"""
        try:
            # Get page content
            content = page.content
            
            # Split into sections
            sections = content.split('\n\n')
            
            # Find sections that contain query terms
            query_terms = set(query.lower().split())
            relevant_sections = []
            
            for section in sections[:10]:  # Limit to first 10 sections
                section_lower = section.lower()
                # Check if section contains any query terms
                if any(term in section_lower for term in query_terms):
                    # Limit section length
                    if len(section) > 500:
                        section = section[:500] + "..."
                    relevant_sections.append(section)
                
                # Stop if we have enough content
                if len(relevant_sections) >= 3:
                    break
            
            return '\n\n'.join(relevant_sections)
            
        except Exception as e:
            logger.warning(f"Failed to extract relevant content: {e}")
            return ""
    
    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """Calculate relevance score based on query term matches"""
        try:
            text_lower = text.lower()
            query_terms = query.lower().split()
            
            if not query_terms:
                return 0.5
            
            # Count term matches
            matches = 0
            for term in query_terms:
                if term in text_lower:
                    matches += 1
            
            # Calculate score (0.3 to 0.9 range for Wikipedia results)
            base_score = 0.3
            match_bonus = (matches / len(query_terms)) * 0.6
            
            return min(0.9, base_score + match_bonus)
            
        except Exception:
            return 0.5
    
    async def get_wikipedia_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """
        Get Wikipedia search suggestions for a query
        
        Args:
            query: Search query
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested search terms
        """
        try:
            clean_query = self._clean_query(query)
            suggestions = wikipedia.search(clean_query, results=limit)
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to get Wikipedia suggestions: {e}")
            return []
    
    async def search_with_fallback_queries(
        self,
        primary_query: str,
        fallback_queries: List[str],
        top_k: int = 3
    ) -> Tuple[List[SearchResult], float]:
        """
        Search Wikipedia with primary query and fallback to alternatives
        
        Args:
            primary_query: Primary search query
            fallback_queries: Alternative queries to try if primary fails
            top_k: Number of results to return
            
        Returns:
            Tuple of (search results, total search time in ms)
        """
        start_time = time.time()
        
        # Try primary query first
        results, _ = await self.search_wikipedia(primary_query, top_k)
        
        # If we don't have enough results, try fallback queries
        if len(results) < top_k and fallback_queries:
            remaining_slots = top_k - len(results)
            
            for fallback_query in fallback_queries:
                if remaining_slots <= 0:
                    break
                
                fallback_results, _ = await self.search_wikipedia(
                    fallback_query, 
                    remaining_slots
                )
                
                # Add unique results (avoid duplicates)
                existing_titles = {r.title for r in results if r.title}
                for result in fallback_results:
                    if result.title and result.title not in existing_titles:
                        results.append(result)
                        existing_titles.add(result.title)
                        remaining_slots -= 1
                        
                        if remaining_slots <= 0:
                            break
        
        total_time = (time.time() - start_time) * 1000
        logger.info(f"Wikipedia fallback search completed in {total_time:.2f}ms")
        
        return results[:top_k], total_time
    
    async def health_check(self) -> bool:
        """Check if Wikipedia service is available"""
        try:
            if not rag_config.enable_wikipedia_fallback:
                return True  # Service is "healthy" if disabled
            
            # Test with a simple search
            test_results = wikipedia.search("test", results=1)
            return len(test_results) >= 0  # Even empty results indicate service is working
            
        except Exception as e:
            logger.error(f"Wikipedia service health check failed: {e}")
            return False


# Global instance
wikipedia_service = WikipediaService()
