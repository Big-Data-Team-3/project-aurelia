"""
Main RAG Service orchestrating all RAG components
Provides high-level interface for RAG operations
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging

from config.rag_config import rag_config, SearchStrategies
from models.rag_models import (
    SearchResult, RAGQuery, RAGResponse, SearchResponse, 
    SearchStrategy, ChatMessage, StreamingChunk
)
from services.vector_search import vector_search_service
from services.generation import generation_service
from services.reranking import reranking_service
from services.wikipedia import wikipedia_service
from services.hybrid_search import hybrid_search_service
from services.fusion import fusion_service


logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service orchestrating retrieval and generation"""
    
    def __init__(self):
        self.services_initialized = False
    
    async def initialize_services(self):
        """Initialize all RAG services"""
        if self.services_initialized:
            return
        
        try:
            logger.info("Initializing RAG services...")
            
            # Initialize BM25 index for hybrid search
            await hybrid_search_service.initialize_bm25_index()
            
            self.services_initialized = True
            logger.info("RAG services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
            raise
    
    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Main RAG query method combining retrieval and generation
        
        Args:
            rag_query: RAG query parameters
            
        Returns:
            Complete RAG response with answer and sources
        """
        start_time = time.time()
        
        try:
            await self.initialize_services()
            
            # Step 1: Retrieve relevant documents
            search_results, retrieval_time = await self._retrieve_documents(rag_query)
            
            # Step 2: Apply Wikipedia fallback if needed
            if (rag_query.enable_wikipedia_fallback and 
                len(search_results) < rag_query.rerank_top_k):
                
                wiki_results, wiki_time = await self._apply_wikipedia_fallback(
                    rag_query.query, 
                    search_results,
                    rag_query.top_k - len(search_results)
                )
                search_results.extend(wiki_results)
                retrieval_time += wiki_time
            
            # Step 3: Generate response
            if search_results:
                answer, generation_time = await generation_service.generate_response(
                    query=rag_query.query,
                    context_results=search_results[:rag_query.rerank_top_k],
                    temperature=rag_query.temperature,
                    max_tokens=rag_query.max_tokens
                )
            else:
                answer = "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic."
                generation_time = 0.0
            
            total_time = (time.time() - start_time) * 1000
            
            # Create response
            response = RAGResponse(
                query=rag_query.query,
                answer=answer,
                sources=search_results[:rag_query.rerank_top_k] if rag_query.include_sources else [],
                strategy_used=rag_query.strategy,
                metadata={
                    'total_sources_found': len(search_results),
                    'sources_used_for_generation': min(len(search_results), rag_query.rerank_top_k),
                    'wikipedia_fallback_used': rag_query.enable_wikipedia_fallback and any(
                        r.source_type.value == 'wikipedia' for r in search_results
                    )
                },
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time
            )
            
            logger.info(f"RAG query completed in {total_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            # Return error response
            return RAGResponse(
                query=rag_query.query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                strategy_used=rag_query.strategy,
                metadata={'error': str(e)},
                total_time_ms=(time.time() - start_time) * 1000
            )
    
    async def search_only(self, query: str, strategy: SearchStrategy, top_k: int = 10) -> SearchResponse:
        """
        Perform search without generation
        
        Args:
            query: Search query
            strategy: Search strategy to use
            top_k: Number of results to return
            
        Returns:
            Search results without generation
        """
        start_time = time.time()
        
        try:
            await self.initialize_services()
            
            # Create RAG query for retrieval
            rag_query = RAGQuery(
                query=query,
                strategy=strategy,
                top_k=top_k,
                rerank_top_k=min(top_k, rag_config.rerank_top_k)
            )
            
            # Retrieve documents
            search_results, retrieval_time = await self._retrieve_documents(rag_query)
            
            return SearchResponse(
                query=query,
                results=search_results,
                strategy_used=strategy,
                total_results=len(search_results),
                retrieval_time_ms=retrieval_time,
            )
            
        except Exception as e:
            logger.error(f"Search-only query failed: {e}")
            return SearchResponse(
                query=query,
                results=[],
                strategy_used=strategy,
                total_results=0,
                retrieval_time_ms=(time.time() - start_time) * 1000
            )
    
    async def streaming_query(self, rag_query: RAGQuery) -> AsyncGenerator[StreamingChunk, None]:
        """
        Streaming RAG query for real-time responses
        
        Args:
            rag_query: RAG query parameters
            
        Yields:
            Streaming chunks with incremental response data
        """
        try:
            await self.initialize_services()
            
            # First yield: Start retrieval
            yield StreamingChunk(
                type="metadata",
                content={"status": "retrieving", "query": rag_query.query},
                is_final=False
            )
            
            # Retrieve documents
            search_results, retrieval_time = await self._retrieve_documents(rag_query)
            
            # Apply Wikipedia fallback if needed
            if (rag_query.enable_wikipedia_fallback and 
                len(search_results) < rag_query.rerank_top_k):
                
                wiki_results, _ = await self._apply_wikipedia_fallback(
                    rag_query.query, 
                    search_results,
                    rag_query.top_k - len(search_results)
                )
                search_results.extend(wiki_results)
            
            # Yield sources
            if rag_query.include_sources:
                yield StreamingChunk(
                    type="sources",
                    content=search_results[:rag_query.rerank_top_k],
                    is_final=False
                )
            
            # Yield generation status
            yield StreamingChunk(
                type="metadata",
                content={"status": "generating", "sources_found": len(search_results)},
                is_final=False
            )
            
            # Stream generation
            if search_results:
                async for text_chunk in generation_service.generate_streaming_response(
                    query=rag_query.query,
                    context_results=search_results[:rag_query.rerank_top_k],
                    temperature=rag_query.temperature,
                    max_tokens=rag_query.max_tokens
                ):
                    yield StreamingChunk(
                        type="text",
                        content=text_chunk,
                        is_final=False
                    )
            else:
                yield StreamingChunk(
                    type="text",
                    content="I couldn't find relevant information to answer your question.",
                    is_final=False
                )
            
            # Final chunk
            yield StreamingChunk(
                type="metadata",
                content={"status": "completed", "retrieval_time_ms": retrieval_time},
                is_final=True
            )
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield StreamingChunk(
                type="text",
                content=f"Error: {str(e)}",
                is_final=True
            )
    
    async def conversational_query(
        self,
        query: str,
        conversation_history: List[ChatMessage],
        strategy: SearchStrategy = SearchStrategy.RRF_FUSION,
        top_k: int = 10
    ) -> RAGResponse:
        """
        RAG query with conversation context
        
        Args:
            query: Current user query
            conversation_history: Previous conversation messages
            strategy: Search strategy to use
            top_k: Number of sources to retrieve
            
        Returns:
            RAG response considering conversation history
        """
        try:
            await self.initialize_services()
            
            # Create RAG query
            rag_query = RAGQuery(
                query=query,
                strategy=strategy,
                top_k=top_k
            )
            
            # Retrieve documents
            search_results, retrieval_time = await self._retrieve_documents(rag_query)
            
            # Generate response with conversation history
            if search_results:
                answer, generation_time = await generation_service.generate_with_conversation_history(
                    query=query,
                    context_results=search_results[:rag_query.rerank_top_k],
                    conversation_history=conversation_history,
                    temperature=rag_query.temperature,
                    max_tokens=rag_query.max_tokens
                )
            else:
                answer = "I couldn't find relevant information to answer your question based on our conversation."
                generation_time = 0.0
            
            return RAGResponse(
                query=query,
                answer=answer,
                sources=search_results[:rag_query.rerank_top_k],
                strategy_used=strategy,
                metadata={
                    'conversation_length': len(conversation_history),
                    'total_sources_found': len(search_results)
                },
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time
            )
            
        except Exception as e:
            logger.error(f"Conversational query failed: {e}")
            return RAGResponse(
                query=query,
                answer=f"Error in conversational query: {str(e)}",
                sources=[],
                strategy_used=strategy,
                metadata={'error': str(e)}
            )
    
    async def _retrieve_documents(self, rag_query: RAGQuery) -> tuple[List[SearchResult], float]:
        """
        Internal method to retrieve documents based on strategy
        
        Args:
            rag_query: RAG query parameters
            
        Returns:
            Tuple of (search results, retrieval time in ms)
        """
        start_time = time.time()
        
        try:
            strategy_config = SearchStrategies.get_strategy_config(rag_query.strategy.value)
            
            if rag_query.strategy == SearchStrategy.VECTOR_ONLY:
                # Pure vector search
                results, _ = await vector_search_service.vector_search(
                    query=rag_query.query,
                    top_k=rag_query.top_k
                )
                
            elif rag_query.strategy == SearchStrategy.HYBRID:
                # Hybrid search without reranking
                results, _ = await hybrid_search_service.hybrid_search(
                    query=rag_query.query,
                    top_k=rag_query.top_k,
                    fusion_method='rrf'
                )
                
            elif rag_query.strategy == SearchStrategy.RERANKED:
                # Vector search + reranking
                vector_results, _ = await vector_search_service.vector_search(
                    query=rag_query.query,
                    top_k=rag_query.top_k * 2  # Get more for reranking
                )
                
                if vector_results and rag_config.enable_reranking:
                    results, _ = await reranking_service.rerank_results(
                        query=rag_query.query,
                        results=vector_results,
                        top_k=rag_query.top_k
                    )
                else:
                    results = vector_results[:rag_query.top_k]
                
            elif rag_query.strategy == SearchStrategy.RRF_FUSION:
                # Full hybrid search with RRF fusion and reranking
                hybrid_results, _ = await hybrid_search_service.hybrid_search(
                    query=rag_query.query,
                    top_k=rag_query.top_k * 2,  # Get more for reranking
                    fusion_method='rrf'
                )
                
                if hybrid_results and rag_config.enable_reranking:
                    results, _ = await reranking_service.rerank_results(
                        query=rag_query.query,
                        results=hybrid_results,
                        top_k=rag_query.top_k
                    )
                else:
                    results = hybrid_results[:rag_query.top_k]
                    
            else:
                # Default to vector search
                results, _ = await vector_search_service.vector_search(
                    query=rag_query.query,
                    top_k=rag_query.top_k
                )
            
            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"Document retrieval completed in {retrieval_time:.2f}ms using {rag_query.strategy.value}")
            
            return results, retrieval_time
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return [], (time.time() - start_time) * 1000
    
    async def _apply_wikipedia_fallback(
        self,
        query: str,
        existing_results: List[SearchResult],
        max_wiki_results: int
    ) -> tuple[List[SearchResult], float]:
        """
        Apply Wikipedia fallback if existing results are insufficient
        
        Args:
            query: Original query
            existing_results: Existing search results
            max_wiki_results: Maximum Wikipedia results to add
            
        Returns:
            Tuple of (Wikipedia results, search time in ms)
        """
        try:
            # Check if we need Wikipedia fallback
            if len(existing_results) >= rag_config.rerank_top_k:
                return [], 0.0
            
            # Check similarity threshold
            if existing_results:
                avg_score = sum(r.score for r in existing_results) / len(existing_results)
                if avg_score >= rag_config.similarity_threshold:
                    return [], 0.0  # Existing results are good enough
            
            # Search Wikipedia
            wiki_results, wiki_time = await wikipedia_service.search_wikipedia(
                query=query,
                top_k=min(max_wiki_results, rag_config.wikipedia_top_k)
            )
            
            logger.info(f"Wikipedia fallback added {len(wiki_results)} results")
            return wiki_results, wiki_time
            
        except Exception as e:
            logger.error(f"Wikipedia fallback failed: {e}")
            return [], 0.0
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Comprehensive health check for all RAG services
        
        Returns:
            Dictionary with health status of each service
        """
        try:
            # Check all services concurrently
            health_checks = await asyncio.gather(
                vector_search_service.health_check(),
                generation_service.health_check(),
                reranking_service.health_check(),
                wikipedia_service.health_check(),
                hybrid_search_service.health_check(),
                fusion_service.health_check(),
                return_exceptions=True
            )
            
            service_names = [
                'vector_search', 'generation', 'reranking', 
                'wikipedia', 'hybrid_search', 'fusion'
            ]
            
            health_status = {}
            for name, result in zip(service_names, health_checks):
                if isinstance(result, Exception):
                    health_status[name] = False
                    logger.error(f"Health check failed for {name}: {result}")
                else:
                    health_status[name] = bool(result)
            
            # Overall health
            health_status['overall'] = all(health_status.values())
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {'overall': False, 'error': str(e)}


# Global instance
rag_service = RAGService()
