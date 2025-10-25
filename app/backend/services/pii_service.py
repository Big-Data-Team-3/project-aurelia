"""
Enhanced PII Detection Service
Handles detection and masking of Personally Identifiable Information in user queries
Uses hybrid approach: regex patterns + ChatGPT for contextual analysis
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from models.rag_models import PIIDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class PIIPattern:
    """Configuration for a PII pattern"""
    name: str
    pattern: str
    replacement: str
    confidence: float
    description: str


class PIIService:
    """Enhanced PII service with ChatGPT backup for contextual analysis"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.enabled = True
        self.strict_mode = False  # If True, reject queries with high-confidence PII
        self.llm_enabled = True
        self.use_llm_for_ambiguous = True
        self._generation_service = None  # Lazy loaded to avoid circular imports
    
    def _initialize_patterns(self) -> Dict[str, PIIPattern]:
        """Initialize PII detection patterns"""
        patterns = {
            "email": PIIPattern(
                name="email",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement="[EMAIL]",
                confidence=0.95,
                description="Email address"
            ),
            "phone_us": PIIPattern(
                name="phone_us",
                pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                replacement="[PHONE]",
                confidence=0.90,
                description="US phone number"
            ),
            "ssn": PIIPattern(
                name="ssn",
                pattern=r'\b(?!000|666|9\d{2})\d{3}[-.\s]?(?!00)\d{2}[-.\s]?(?!0000)\d{4}\b',
                replacement="[SSN]",
                confidence=0.95,
                description="Social Security Number"
            ),
            "credit_card": PIIPattern(
                name="credit_card",
                pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                replacement="[CREDIT_CARD]",
                confidence=0.90,
                description="Credit card number"
            ),
            "ip_address": PIIPattern(
                name="ip_address",
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                replacement="[IP_ADDRESS]",
                confidence=0.80,
                description="IP address"
            ),
            "url_with_personal": PIIPattern(
                name="url_with_personal",
                pattern=r'https?://[^\s]*(?:user|profile|account|personal)[^\s]*',
                replacement="[PERSONAL_URL]",
                confidence=0.70,
                description="URL with personal information"
            ),
            "address_pattern": PIIPattern(
                name="address_pattern",
                pattern=r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
                replacement="[ADDRESS]",
                confidence=0.75,
                description="Street address"
            ),
            "date_of_birth": PIIPattern(
                name="date_of_birth",
                pattern=r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b',
                replacement="[DATE_OF_BIRTH]",
                confidence=0.85,
                description="Date of birth (MM/DD/YYYY or MM-DD-YYYY)"
            )
        }
        
        return patterns
    
    async def detect_pii(self, text: str, confidence_threshold: float = 0.7) -> PIIDetectionResult:
        """
        Enhanced PII detection with ChatGPT fallback for contextual analysis
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence score to consider a match
            
        Returns:
            PIIDetectionResult with detection details
        """
        if not self.enabled:
            return PIIDetectionResult(
                has_pii=False,
                pii_types=[],
                masked_query=text,
                confidence_score=0.0,
                detected_patterns=[]
            )
        
        # Step 1: Run regex-based detection
        regex_result = await self._regex_detect_pii(text, confidence_threshold)
        
        # Step 2: Use ChatGPT for ambiguous cases or as verification
        if self.llm_enabled and self._should_use_llm(regex_result, text):
            try:
                llm_result = await self._llm_pii_detection(text, regex_result)
                return self._merge_results(regex_result, llm_result)
            except Exception as e:
                logger.error(f"LLM PII detection failed, falling back to regex: {e}")
        
        return regex_result
    
    async def _regex_detect_pii(self, text: str, confidence_threshold: float = 0.7) -> PIIDetectionResult:
        """
        Original regex-based PII detection (moved from main detect_pii method)
        """
        detected_patterns = []
        pii_types = []
        masked_text = text
        max_confidence = 0.0
        
        # Check each pattern
        for pattern_name, pattern_config in self.patterns.items():
            matches = list(re.finditer(pattern_config.pattern, text, re.IGNORECASE))
            
            if matches:
                for match in matches:
                    if pattern_config.confidence >= confidence_threshold:
                        # Record the detection
                        detected_patterns.append({
                            "type": pattern_name,
                            "text": match.group(),
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": pattern_config.confidence,
                            "description": pattern_config.description,
                            "source": "regex"
                        })
                        
                        if pattern_name not in pii_types:
                            pii_types.append(pattern_name)
                        
                        max_confidence = max(max_confidence, pattern_config.confidence)
        
        # Mask PII in text
        if detected_patterns:
            # Sort by position (reverse order to maintain indices)
            detected_patterns.sort(key=lambda x: x["start"], reverse=True)
            
            for detection in detected_patterns:
                pattern_config = self.patterns[detection["type"]]
                start, end = detection["start"], detection["end"]
                masked_text = masked_text[:start] + pattern_config.replacement + masked_text[end:]
        
        has_pii = len(detected_patterns) > 0
        
        # Log PII detection for compliance
        if has_pii:
            logger.debug(f"Regex PII detected: {len(detected_patterns)} patterns found, types: {pii_types}")
        
        return PIIDetectionResult(
            has_pii=has_pii,
            pii_types=pii_types,
            masked_query=masked_text,
            confidence_score=max_confidence,
            detected_patterns=detected_patterns
        )
    
    def _should_use_llm(self, regex_result: PIIDetectionResult, text: str) -> bool:
        """Determine when to use ChatGPT for PII detection"""
        # Use LLM when:
        # 1. Regex found potential PII but confidence is low
        # 2. Text contains financial terms that might be confused with PII
        # 3. Text has patterns that look like examples/hypotheticals
        
        financial_indicators = [
            "portfolio", "bond", "investment", "example", "suppose", "assume",
            "hypothetical", "let's say", "consider", "imagine", "what if",
            "for instance", "sample", "demo", "test", "mock"
        ]
        has_financial_context = any(indicator in text.lower() for indicator in financial_indicators)
        
        low_confidence_detections = [
            p for p in regex_result.detected_patterns 
            if p["confidence"] < 0.85
        ]
        
        return (
            (regex_result.has_pii and has_financial_context) or  # Financial context with PII
            len(low_confidence_detections) > 0 or  # Low confidence detections
            self._has_ambiguous_patterns(text)  # Ambiguous patterns
        )
    
    def _has_ambiguous_patterns(self, text: str) -> bool:
        """Check for patterns that might be ambiguous (numbers that could be financial vs PII)"""
        # Look for patterns that could be financial data vs PII
        ambiguous_patterns = [
            r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # Could be SSN or account number
            r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b',  # Could be credit card or account
            r'\b\d{10,}\b',  # Long numbers could be various things
        ]
        
        for pattern in ambiguous_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _get_generation_service(self):
        """Lazy load generation service to avoid circular imports"""
        if self._generation_service is None:
            try:
                from services.generation import generation_service
                self._generation_service = generation_service
            except ImportError:
                logger.error("Could not import generation service for LLM PII detection")
                return None
        return self._generation_service
    
    async def _llm_pii_detection(self, text: str, regex_result: PIIDetectionResult) -> Dict[str, Any]:
        """Use ChatGPT for contextual PII analysis with enhanced error handling"""
        
        generation_service = self._get_generation_service()
        if not generation_service:
            logger.warning("Generation service not available for LLM PII detection")
            return {"has_real_pii": regex_result.has_pii, "confidence": 0.5, "error": "service_unavailable"}
        
        # Check if API key is configured
        if not hasattr(generation_service.client, 'api_key') or not generation_service.client.api_key:
            logger.warning("OpenAI API key not configured, skipping LLM PII analysis")
            return {"has_real_pii": regex_result.has_pii, "confidence": 0.5, "error": "api_key_missing"}
        
        prompt = f"""Analyze the following text for Personally Identifiable Information (PII). 

Text: "{text}"

Context: This is from a financial analysis system where users ask about investments, portfolios, and financial calculations.

Consider:
1. Is this real PII or hypothetical/example data?
2. Are numbers financial data (amounts, rates, IDs) or personal identifiers?
3. Does the context suggest this is educational/example content?

Current regex detections: {[p["type"] for p in regex_result.detected_patterns]}

Respond in JSON format:
{{
    "has_real_pii": boolean,
    "pii_types": ["type1", "type2"],
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "false_positives": ["regex_type1", "regex_type2"],
    "missed_pii": [
        {{"type": "type", "text": "detected_text", "confidence": 0.0-1.0}}
    ]
}}"""

        try:
            # The OpenAI client is synchronous, not async
            response = generation_service.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for PII detection
                messages=[
                    {"role": "system", "content": "You are an expert at detecting PII while understanding context and avoiding false positives in financial/educational content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                logger.warning("Empty response from OpenAI API")
                return {"has_real_pii": regex_result.has_pii, "confidence": 0.5, "error": "empty_response"}
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {response_content}")
            
            result = json.loads(response_content)
            logger.debug(f"LLM PII analysis: {result.get('reasoning', 'No reasoning provided')}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM PII response as JSON: {e}")
            logger.error(f"Raw response content: '{response_content if 'response_content' in locals() else 'N/A'}'")
            return {"has_real_pii": regex_result.has_pii, "confidence": 0.5, "error": "json_parse_failed"}
        except AttributeError as e:
            logger.error(f"OpenAI client attribute error (likely API key issue): {e}")
            return {"has_real_pii": regex_result.has_pii, "confidence": 0.5, "error": "client_config_error"}
        except Exception as e:
            logger.error(f"LLM PII detection failed: {e}")
            return {"has_real_pii": regex_result.has_pii, "confidence": 0.5, "error": str(e)}
    
    def _merge_results(self, regex_result: PIIDetectionResult, llm_result: Dict[str, Any]) -> PIIDetectionResult:
        """Merge regex and LLM results intelligently"""
        
        # Check if LLM had an error
        if "error" in llm_result:
            logger.info(f"LLM PII analysis had error: {llm_result['error']}, using regex results only")
            return regex_result
        
        # Remove false positives identified by LLM
        false_positives = llm_result.get("false_positives", [])
        filtered_patterns = [
            p for p in regex_result.detected_patterns 
            if p["type"] not in false_positives
        ]
        
        # Add missed PII identified by LLM
        missed_pii = llm_result.get("missed_pii", [])
        for missed in missed_pii:
            filtered_patterns.append({
                "type": missed["type"],
                "text": missed["text"],
                "start": -1,  # LLM doesn't provide position
                "end": -1,
                "confidence": missed["confidence"],
                "source": "llm",
                "description": f"LLM-detected {missed['type']}"
            })
        
        # Update masked query
        masked_query = regex_result.masked_query
        for missed in missed_pii:
            masked_query = masked_query.replace(missed["text"], f"[{missed['type'].upper()}]")
        
        # Determine final result
        has_pii = llm_result.get("has_real_pii", len(filtered_patterns) > 0)
        final_confidence = max(
            llm_result.get("confidence", 0.5),
            max([p["confidence"] for p in filtered_patterns], default=0.0)
        )
        
        # Log the decision
        if has_pii != regex_result.has_pii:
            logger.info(f"LLM overrode regex PII detection: regex={regex_result.has_pii}, llm={has_pii}, reasoning={llm_result.get('reasoning', 'N/A')}")
        elif false_positives:
            logger.info(f"LLM filtered out false positives: {false_positives}")
        
        return PIIDetectionResult(
            has_pii=has_pii,
            pii_types=list(set([p["type"] for p in filtered_patterns])),
            masked_query=masked_query,
            confidence_score=final_confidence,
            detected_patterns=filtered_patterns
        )
    
    async def should_reject_query(self, pii_result: PIIDetectionResult, strict_threshold: float = 0.9) -> Tuple[bool, str]:
        """
        Determine if a query should be rejected based on PII detection
        
        Args:
            pii_result: PII detection result
            strict_threshold: Confidence threshold for rejection
            
        Returns:
            Tuple of (should_reject, reason)
        """
        if not self.strict_mode:
            return False, ""
        
        if not pii_result.has_pii:
            return False, ""
        
        # Check for high-confidence sensitive PII
        sensitive_types = ["ssn", "credit_card", "date_of_birth"]
        high_confidence_sensitive = [
            p for p in pii_result.detected_patterns
            if p["type"] in sensitive_types and p["confidence"] >= strict_threshold
        ]
        
        if high_confidence_sensitive:
            types_found = [p["type"] for p in high_confidence_sensitive]
            return True, f"Query contains sensitive PII: {', '.join(types_found)}"
        
        # Check for multiple PII types (potential identity exposure)
        if len(pii_result.pii_types) >= 3 and pii_result.confidence_score >= strict_threshold:
            return True, f"Query contains multiple PII types: {', '.join(pii_result.pii_types)}"
        
        return False, ""
    
    def add_custom_pattern(self, name: str, pattern: str, replacement: str, confidence: float, description: str):
        """
        Add a custom PII pattern
        
        Args:
            name: Pattern name
            pattern: Regex pattern
            replacement: Replacement text
            confidence: Confidence score (0-1)
            description: Pattern description
        """
        self.patterns[name] = PIIPattern(
            name=name,
            pattern=pattern,
            replacement=replacement,
            confidence=confidence,
            description=description
        )
        logger.info(f"Added custom PII pattern: {name}")
    
    def remove_pattern(self, name: str) -> bool:
        """
        Remove a PII pattern
        
        Args:
            name: Pattern name to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        if name in self.patterns:
            del self.patterns[name]
            logger.info(f"Removed PII pattern: {name}")
            return True
        return False
    
    def get_pattern_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all configured patterns
        
        Returns:
            Dictionary with pattern information
        """
        return {
            name: {
                "description": pattern.description,
                "confidence": pattern.confidence,
                "replacement": pattern.replacement
            }
            for name, pattern in self.patterns.items()
        }
    
    def configure(self, enabled: bool = None, strict_mode: bool = None, llm_enabled: bool = None, use_llm_for_ambiguous: bool = None):
        """
        Configure PII service settings
        
        Args:
            enabled: Enable/disable PII detection
            strict_mode: Enable/disable strict mode (query rejection)
            llm_enabled: Enable/disable LLM-enhanced PII detection
            use_llm_for_ambiguous: Enable/disable LLM for ambiguous cases only
        """
        if enabled is not None:
            self.enabled = enabled
            logger.info(f"PII detection {'enabled' if enabled else 'disabled'}")
        
        if strict_mode is not None:
            self.strict_mode = strict_mode
            logger.info(f"PII strict mode {'enabled' if strict_mode else 'disabled'}")
        
        if llm_enabled is not None:
            self.llm_enabled = llm_enabled
            logger.info(f"LLM-enhanced PII detection {'enabled' if llm_enabled else 'disabled'}")
        
        if use_llm_for_ambiguous is not None:
            self.use_llm_for_ambiguous = use_llm_for_ambiguous
            logger.info(f"LLM for ambiguous PII cases {'enabled' if use_llm_for_ambiguous else 'disabled'}")
    
    async def health_check(self) -> bool:
        """
        Check if PII service is healthy
        
        Returns:
            True if service is healthy
        """
        try:
            # Test with a sample text
            test_result = await self.detect_pii("test@example.com")
            return test_result.has_pii and "email" in test_result.pii_types
        except Exception as e:
            logger.error(f"PII service health check failed: {e}")
            return False
    
    def get_llm_status(self) -> Dict[str, Any]:
        """
        Get detailed status of LLM PII detection capabilities
        
        Returns:
            Dictionary with LLM status information
        """
        status = {
            "llm_enabled": self.llm_enabled,
            "use_llm_for_ambiguous": self.use_llm_for_ambiguous,
            "generation_service_available": False,
            "api_key_configured": False,
            "status": "unknown"
        }
        
        try:
            generation_service = self._get_generation_service()
            if generation_service:
                status["generation_service_available"] = True
                
                # Check API key
                if hasattr(generation_service.client, 'api_key') and generation_service.client.api_key:
                    status["api_key_configured"] = True
                    status["status"] = "ready"
                else:
                    status["status"] = "api_key_missing"
            else:
                status["status"] = "service_unavailable"
                
        except Exception as e:
            status["status"] = f"error: {str(e)}"
        
        return status


# Global PII service instance
pii_service = PIIService()
