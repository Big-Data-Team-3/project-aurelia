"""
Main RAG Service orchestrating all RAG components
Provides high-level interface for RAG operations
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import logging
import re
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from config.rag_config import rag_config, SearchStrategies
from models.rag_models import (
    SearchResult, RAGQuery, RAGResponse, SearchResponse, 
    SearchStrategy, ChatMessage, StreamingChunk, ChatSession,
    SessionMessageRequest, SessionMessageResponse, QueryProcessingResult
)
from services.vector_search import vector_search_service
from services.generation import generation_service
from services.reranking import reranking_service
from services.wikipedia import wikipedia_service
from services.hybrid_search import hybrid_search_service
from services.fusion import fusion_service
from services.query_processor import query_processor
from services.chatgpt_service import chatgpt_service
from services.session_service import session_service
from services.query_rewriter import query_rewriter
from services.context_manager import context_manager
from services.query_classifier import query_classifier, QueryType
from services.instructor_classifier import instructor_classifier
from models.instructor_models import QueryType as InstructorQueryType


logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service orchestrating retrieval and generation"""
    
    def __init__(self):
        self.services_initialized = False
        # Concurrent request limiting for financial enhancement
        self.enhancement_semaphore = asyncio.Semaphore(rag_config.max_concurrent_enhancements)
    
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
    
    async def session_message(
        self, 
        session_id: str, 
        message_request: SessionMessageRequest
    ) -> SessionMessageResponse:
        """
        Process a message within a session context with full query processing pipeline
        
        Args:
            session_id: Session identifier
            message_request: Message request with query and parameters
            
        Returns:
            SessionMessageResponse with processed message and response
        """
        start_time = time.time()
        
        try:
            await self.initialize_services()
            
            # Get session
            session = await session_service.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found or expired")
            
            # Step 1: Get optimized context using context manager
            context_window = await context_manager.get_optimized_context(session, message_request.message)
            conversation_history = context_window.messages
            
            query_processing_result = await query_processor.process_query(
                message_request.message, 
                conversation_history
            )
            
            # Create user message
            user_message = ChatMessage(
                role="user",
                content=query_processing_result.processed_query
            )
            
            # Step 2: Route to RAG or ChatGPT based on processing result
            if query_processing_result.routing_decision == "RAG":
                assistant_response, sources = await self._process_rag_query(
                    query_processing_result.processed_query,
                    conversation_history,
                    message_request.strategy,
                    message_request.temperature,
                    message_request.max_tokens,
                    query_processing_result.expanded_queries
                )
            else:
                assistant_response, _ = await chatgpt_service.generate_response(
                    query_processing_result.processed_query,
                    conversation_history,
                    message_request.temperature,
                    message_request.max_tokens
                )
                sources = []
            
            # Create assistant message
            assistant_message = ChatMessage(
                role="assistant",
                content=assistant_response,
                sources=sources
            )
            
            # Step 3: Update session with both messages
            await session_service.update_session(session_id, user_message)
            await session_service.update_session(session_id, assistant_message)
            
            total_time = (time.time() - start_time) * 1000
            
            return SessionMessageResponse(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                query_processing={
                    "pii_detected": query_processing_result.pii_detection.has_pii,
                    "pii_types": query_processing_result.pii_detection.pii_types,
                    "intent": query_processing_result.intent_classification["predicted_intent"],
                    "confidence": query_processing_result.intent_classification["confidence_score"],
                    "expanded_queries": query_processing_result.expanded_queries,
                    "processing_time_ms": query_processing_result.processing_time_ms
                },
                routing_decision=query_processing_result.routing_decision,
                sources=sources,
                total_time_ms=total_time
            )
            
        except Exception as e:
            logger.error(f"Session message processing failed: {e}")
            # Still try to save user message to session
            try:
                user_message = ChatMessage(role="user", content=message_request.message)
                await session_service.update_session(session_id, user_message)
                
                error_message = ChatMessage(
                    role="assistant", 
                    content="I apologize, but I encountered an error processing your message. Please try again."
                )
                await session_service.update_session(session_id, error_message)
                
                total_time = (time.time() - start_time) * 1000
                
                return SessionMessageResponse(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_message=error_message,
                    query_processing={"error": str(e)},
                    routing_decision="error",
                    sources=[],
                    total_time_ms=total_time
                )
            except Exception as session_error:
                logger.error(f"Failed to save error to session: {session_error}")
                raise e
    
    async def _process_rag_query(
        self,
        query: str,
        conversation_history: List[ChatMessage],
        strategy: SearchStrategy,
        temperature: float,
        max_tokens: int,
        expanded_queries: List[str] = None
    ) -> tuple[str, List[SearchResult]]:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: Processed query text
            conversation_history: Previous conversation messages
            strategy: Search strategy to use
            temperature: Generation temperature
            max_tokens: Maximum tokens
            expanded_queries: Additional query variations
            
        Returns:
            Tuple of (response text, source results)
        """
        # Create RAG query
        rag_query = RAGQuery(
            query=query,
            strategy=strategy,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Retrieve documents (potentially using expanded queries)
        search_results, _ = await self._retrieve_documents(rag_query)
        
        # If we have expanded queries and few results, try them
        if expanded_queries and len(search_results) < 3:
            for expanded_query in expanded_queries[:2]:  # Try top 2 expansions
                expanded_rag_query = RAGQuery(
                    query=expanded_query,
                    strategy=strategy,
                    top_k=5  # Fewer results per expansion
                )
                additional_results, _ = await self._retrieve_documents(expanded_rag_query)
                search_results.extend(additional_results)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_results = []
        for result in search_results:
            if result.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.id)
        
        search_results = unique_results[:rag_query.rerank_top_k]
        
        # Generate response with conversation history
        if search_results:
            answer, _ = await generation_service.generate_with_conversation_history(
                query=query,
                context_results=search_results,
                conversation_history=conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            answer = "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic."
        
        return answer, search_results
    
    async def _handle_session_management(self, rag_query: RAGQuery) -> Optional[ChatSession]:
        """Smart session management with user-based fallback"""
        try:
            session = None
            user_id = rag_query.user_id or "anonymous"
            
            # Strategy 1: Use provided session_id
            if rag_query.session_id:
                session = await session_service.get_session(rag_query.session_id)
                if session:
                    logger.info(f"Using provided session {session.session_id}")
                    return session
                else:
                    logger.warning(f"Provided session {rag_query.session_id} not found")
            
            # Strategy 2: Find existing user session if no session_id or session not found
            if not session and rag_query.create_session:
                user_sessions = await session_service.get_user_sessions(user_id, active_only=True)
                
                if user_sessions:
                    # Use most recent session (already sorted by updated_at desc)
                    session = user_sessions[0]
                    logger.info(f"Using existing user session {session.session_id}")
                else:
                    # Strategy 3: Create new session only if no existing sessions
                    session = await session_service.create_session(user_id)
                    logger.info(f"Created new session {session.session_id} for user {user_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"Session management failed: {e}")
            return None

    async def _update_session_with_conversation(
        self, 
        session: ChatSession, 
        user_query: str, 
        assistant_answer: str, 
        sources: List[SearchResult]
    ):
        """Update session with new conversation"""
        try:
            # Add user message
            user_message = ChatMessage(
                role="user",
                content=user_query,
                timestamp=datetime.now()
            )
            await session_service.update_session(session.session_id, user_message)
            
            # Add assistant message
            assistant_message = ChatMessage(
                role="assistant",
                content=assistant_answer,
                sources=sources,
                timestamp=datetime.now()
            )
            await session_service.update_session(session.session_id, assistant_message)
            
            logger.debug(f"Updated session {session.session_id} with new conversation")
            
        except Exception as e:
            logger.error(f"Failed to update session with conversation: {e}")

    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Enhanced RAG query method with intelligent query classification and routing
        """
        start_time = time.time()
        
        try:
            print(f"--------------------------------")
            await self.initialize_services()
            logger.info(f"Query: {rag_query.query}")
            
            # Step 1: Handle session management
            session = await self._handle_session_management(rag_query)
            
            # Step 2: Get conversation history from session or query metadata
            conversation_history = []
            if session:
                context_window = await context_manager.get_optimized_context(session, rag_query.query)
                conversation_history = context_window.messages
                logger.info(f"Using session {session.session_id} with {len(conversation_history)} context messages")
            elif rag_query.metadata and 'conversation_context' in rag_query.metadata:
                # Use conversation context from query metadata
                conversation_history = rag_query.metadata['conversation_context']
                logger.info(f"Using conversation context from metadata with {len(conversation_history)} messages")
            
            # Step 3: Classify query type using Instructor with fallback
            try:
                classification = await instructor_classifier.classify_query(rag_query.query, conversation_history)
                query_type = classification.query_type
                logger.info(f"Query classified as: {query_type.value} (confidence: {classification.confidence:.2f})")
            except Exception as e:
                logger.warning(f"Instructor classification failed, using fallback: {e}")
                query_type = await query_classifier.classify_query(rag_query.query, conversation_history)
                logger.info(f"Fallback classification: {query_type.value}")
            
            # Step 4: Route to appropriate handler
            if query_type == QueryType.EXTERNAL:
                response = await self._handle_external_query(rag_query, conversation_history)
            elif query_type == QueryType.CONTEXT_DEPENDENT:
                response = await self._handle_context_dependent_query(rag_query, conversation_history)
            else:  # DOCUMENT_BASED
                response = await self._handle_document_based_query(rag_query, conversation_history)
            
            # Step 5: Update session with conversation if session exists
            if session:
                await self._update_session_with_conversation(session, rag_query.query, response.answer, response.sources)
                response.session_id = session.session_id
                response.message_count = len(session.messages)
            
            # Step 6: Add classification info to metadata
            response.metadata['query_classification'] = query_type.value
            if 'classification' in locals():
                # Instructor classification was used
                response.metadata['classification_confidence'] = classification.confidence
                response.metadata['classification_reasoning'] = classification.reasoning
                response.metadata['classification_method'] = 'instructor'
            else:
                # Fallback classification was used
                response.metadata['classification_reason'] = query_classifier.get_classification_reason(rag_query.query, query_type)
                response.metadata['classification_method'] = 'fallback'
            
            total_time = (time.time() - start_time) * 1000
            response.total_time_ms = total_time
            
            logger.info(f"RAG query completed in {total_time:.2f}ms with classification: {query_type.value}")
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
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
        RAG query with optimized conversation context
        
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
            
            # Create a temporary session for context optimization
            temp_session = ChatSession(
                session_id="temp",
                user_id="temp",
                messages=conversation_history,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            
            # Get optimized context
            context_window = await context_manager.get_optimized_context(temp_session, query)
            optimized_history = context_window.messages
            
            # Create RAG query
            rag_query = RAGQuery(
                query=query,
                strategy=strategy,
                top_k=top_k
            )
            
            # Retrieve documents
            search_results, retrieval_time = await self._retrieve_documents(rag_query)
            
            # Generate response with optimized conversation history
            if search_results:
                answer, generation_time = await generation_service.generate_with_conversation_history(
                    query=query,
                    context_results=search_results[:rag_query.rerank_top_k],
                    conversation_history=optimized_history,
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
                    'original_conversation_length': len(conversation_history),
                    'optimized_conversation_length': len(optimized_history),
                    'context_strategy': context_window.window_type,
                    'context_tokens': context_window.token_count,
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
        max_wiki_results: int = rag_config.wikipedia_top_k
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
            print(f"Applying Wikipedia fallback with query: {query}")
    
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
    

    async def _handle_external_query(
        self, 
        rag_query: RAGQuery, 
        conversation_history: List[ChatMessage]
    ) -> RAGResponse:
        """Handle external queries using conversation context only"""
        try:
            logger.info("Processing external query with conversation context only")
            
            # Use conversation-aware generation without document context
            answer, generation_time = await generation_service.generate_with_conversation_history(
                query=rag_query.query,
                context_results=[],  # No document context
                conversation_history=conversation_history,
                temperature=rag_query.temperature,
                max_tokens=rag_query.max_tokens
            )
            
            # Create response with no sources
            response = RAGResponse(
                query=rag_query.query,
                answer=answer,
                sources=[],  # No sources for external queries
                strategy_used=rag_query.strategy,
                session_id=None,  # Will be set by caller
                message_count=len(conversation_history),
                context_strategy="external",
                metadata={
                    'query_type': 'external',
                    'processing_method': 'conversation_context_only',
                    'sources_used': 0,
                    'document_search_performed': False,
                    'wikipedia_fallback_used': False
                },
                retrieval_time_ms=0.0,  # No retrieval for external queries
                generation_time_ms=generation_time,
                total_time_ms=generation_time
            )
            
            logger.info(f"External query processed in {generation_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"External query processing failed: {e}")
            return RAGResponse(
                query=rag_query.query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                strategy_used=rag_query.strategy,
                metadata={'error': str(e), 'query_type': 'external'},
                total_time_ms=0.0
            )

    async def _handle_context_dependent_query(
        self, 
        rag_query: RAGQuery, 
        conversation_history: List[ChatMessage]
    ) -> RAGResponse:
        """Handle context-dependent queries with enhanced context"""
        try:
            logger.info("Processing context-dependent query with enhanced context")
            
            # Extract context from conversation history
            context_enhanced_query = self._enhance_query_with_context(
                rag_query.query, 
                conversation_history
            )
            
            # Create enhanced RAG query
            enhanced_rag_query = RAGQuery(
                query=context_enhanced_query,
                strategy=rag_query.strategy,
                top_k=rag_query.top_k,
                rerank_top_k=rag_query.rerank_top_k,
                include_sources=rag_query.include_sources,
                enable_wikipedia_fallback=rag_query.enable_wikipedia_fallback,
                temperature=rag_query.temperature,
                max_tokens=rag_query.max_tokens
            )
            
            # Retrieve documents with enhanced query
            search_results, retrieval_time = await self._retrieve_documents(enhanced_rag_query)
            
            # Generate response with both document context and conversation history
            if search_results:
                answer, generation_time = await generation_service.generate_with_conversation_history(
                    query=rag_query.query,  # Use original query for generation
                    context_results=search_results,
                    conversation_history=conversation_history,
                    temperature=rag_query.temperature,
                    max_tokens=rag_query.max_tokens
                )
            else:
                answer = "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic."
                generation_time = 0.0
            
            # Create response
            response = RAGResponse(
                query=rag_query.query,
                answer=answer,
                sources=search_results if rag_query.include_sources else [],
                strategy_used=rag_query.strategy,
                session_id=None,  # Will be set by caller
                message_count=len(conversation_history),
                context_strategy="context_enhanced",
                metadata={
                    'query_type': 'context_dependent',
                    'original_query': rag_query.query,
                    'enhanced_query': context_enhanced_query,
                    'processing_method': 'enhanced_rag_with_context',
                    'sources_used': len(search_results),
                    'document_search_performed': True,
                    'wikipedia_fallback_used': False
                },
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=retrieval_time + generation_time
            )
            
            logger.info(f"Context-dependent query processed in {response.total_time_ms:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Context-dependent query processing failed: {e}")
            return RAGResponse(
                query=rag_query.query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                strategy_used=rag_query.strategy,
                metadata={'error': str(e), 'query_type': 'context_dependent'},
                total_time_ms=0.0
            )

    def _enhance_query_with_context(self, query: str, conversation_history: List[ChatMessage]) -> str:
        """Enhance query with context from conversation history"""
        try:
            # Extract key terms from recent conversation
            recent_messages = conversation_history[-5:]  # Last 5 messages
            context_terms = []
            
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                else:
                    content = msg.get('content', '')
                
                # Extract financial terms, ratios, concepts
                financial_terms = re.findall(r'\b[A-Z][a-z]+\s+(?:Ratio|Rate|Index|Model|Analysis|Risk|Return)\b', content)
                context_terms.extend(financial_terms)
            
            # Remove duplicates and limit
            context_terms = list(set(context_terms))[:3]
            
            # Enhance query
            if context_terms:
                enhanced_query = f"{query} {', '.join(context_terms)}"
                logger.info(f"Enhanced query: '{query}' -> '{enhanced_query}'")
                return enhanced_query
            
            return query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query
            
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

    async def _handle_document_based_query(
        self, 
        rag_query: RAGQuery, 
        conversation_history: List[ChatMessage]
    ) -> RAGResponse:
        """Handle document-based queries with standard RAG + Wikipedia fallback"""
        try:
            logger.info("Processing document-based query with standard RAG pipeline")
            
            # Step 1: Rewrite query for better retrieval
            rewritten_query = await query_rewriter.rewrite_query(rag_query.query)
            
            # Step 2: Create retrieval query with rewritten query
            retrieval_query = RAGQuery(
                query=rewritten_query,  # Use rewritten query for retrieval
                strategy=rag_query.strategy,
                top_k=rag_query.top_k,
                rerank_top_k=rag_query.rerank_top_k,
                include_sources=rag_query.include_sources,
                enable_wikipedia_fallback=rag_query.enable_wikipedia_fallback,
                temperature=rag_query.temperature,
                max_tokens=rag_query.max_tokens
            )
            
            # Step 3: Retrieve documents
            search_results, retrieval_time = await self._retrieve_documents(retrieval_query)
            
            # Step 4: Check confidence and apply Wikipedia fallback if needed
            confidence_threshold = 0.5
            avg_confidence = sum(result.score for result in search_results) / len(search_results) if search_results else 0.0
            
            if (rag_query.enable_wikipedia_fallback and 
                (len(search_results) < rag_query.rerank_top_k or avg_confidence < confidence_threshold)):
                
                logger.info(f"Applying Wikipedia fallback because of insufficient results with avg confidence of {avg_confidence:.2f}")
                wiki_results, wiki_time = await self._apply_wikipedia_fallback(rag_query.query)
                print(f"-----------------")
                print(f"Wikipedia fallback results: {wiki_results}")
                print(f"-----------------")
                search_results = wiki_results
                retrieval_time += wiki_time
            
            # Step 5: Remove duplicates and limit results
            seen_ids = set()
            unique_results = []
            for result in search_results:
                if result.id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result.id)
            
            search_results = unique_results[:rag_query.rerank_top_k]
            
            # Step 6: Generate response
            if search_results:
                if conversation_history:
                    # Use conversation-aware generation
                    answer, generation_time = await generation_service.generate_with_conversation_history(
                        query=rag_query.query,
                        context_results=search_results,
                        conversation_history=conversation_history,
                        temperature=rag_query.temperature,
                        max_tokens=rag_query.max_tokens
                    )
                else:
                    # Use standard generation
                    answer, generation_time = await generation_service.generate_response(
                        query=rag_query.query,
                        context_results=search_results,
                        temperature=rag_query.temperature,
                        max_tokens=rag_query.max_tokens
                    )
            else:
                answer = "I couldn't find relevant information to answer your question. Please try rephrasing your query or asking about a different topic."
                generation_time = 0.0
            
            # Step 6.5: Enhance response with structured financial output
            enhanced_answer = answer
            structured_metadata = {}
            
            if (answer and not answer.startswith("I couldn't find relevant information") 
                and rag_config.enable_financial_enhancement):
                try:
                    logger.info("Enhancing response with structured financial output...")
                    
                    # Use semaphore to limit concurrent enhancements
                    async with self.enhancement_semaphore:
                        # Add configurable delay to prevent overwhelming the API
                        await asyncio.sleep(rag_config.financial_enhancement_delay)
                        
                        financial_response = await instructor_classifier.enhance_financial_response(
                            original_answer=answer,
                            query=rag_query.query,
                            sources=search_results
                        )
                    
                    # Use enhanced answer
                    enhanced_answer = financial_response.answer
                    
                    # Add structured metadata
                    structured_metadata = {
                        'financial_enhancement': True,
                        'formulas_found': financial_response.formulas,
                        'key_metrics': financial_response.key_metrics,
                        'response_confidence': financial_response.confidence,
                        'follow_up_questions': financial_response.follow_up_questions,
                        'risk_level': financial_response.risk_level,
                        'time_horizon': financial_response.time_horizon,
                        'matlab_code': financial_response.matlab_code
                    }
                    
                    logger.info(f"Response enhanced with {len(financial_response.formulas)} formulas and {len(financial_response.key_metrics)} metrics")
                    
                except Exception as e:
                    logger.warning(f"Financial response enhancement failed: {e}")
                    structured_metadata = {'financial_enhancement': False, 'enhancement_error': str(e)}
            
            # Step 7: Create response
            response = RAGResponse(
                query=rag_query.query,
                answer=enhanced_answer,  # Use enhanced answer
                sources=search_results if rag_query.include_sources else [],
                strategy_used=rag_query.strategy,
                session_id=None,  # Will be set by caller
                message_count=len(conversation_history),
                context_strategy="document_based",
                metadata={
                    'query_type': 'document_based',
                    'original_query': rag_query.query,
                    'rewritten_query': rewritten_query,
                    'processing_method': 'standard_rag_with_wikipedia_fallback',
                    'total_sources_found': len(search_results),
                    'sources_used_for_generation': len(search_results),
                    'average_confidence': avg_confidence,
                    'wikipedia_fallback_used': any(
                        r.source_type.value == 'wikipedia' for r in search_results
                    ),
                    'query_rewriting_enabled': True,
                    'document_search_performed': True,
                    **structured_metadata  # Include structured financial metadata
                },
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=retrieval_time + generation_time
            )
            
            logger.info(f"Document-based query processed in {response.total_time_ms:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Document-based query processing failed: {e}")
            return RAGResponse(
                query=rag_query.query,
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                strategy_used=rag_query.strategy,
                metadata={'error': str(e), 'query_type': 'document_based'},
                total_time_ms=0.0
            )


# Global instance
rag_service = RAGService()
