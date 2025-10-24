"""
Minimal Instructor Models
Only for query classification
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


class QueryType(str, Enum):
    """Query classification types"""
    EXTERNAL = "external"
    CONTEXT_DEPENDENT = "context_dependent"
    DOCUMENT_BASED = "document_based"


class QueryClassification(BaseModel):
    """Minimal query classification using Instructor"""
    query_type: QueryType = Field(description="The type of query based on analysis")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation for the classification decision")


class FinancialResponse(BaseModel):
    """
    Structured financial response with enhanced information.
    """
    answer: str = Field(..., description="Main answer to the financial query")
    formulas: List[str] = Field(default=[], description="Mathematical formulas found in the response")
    key_metrics: List[str] = Field(default=[], description="Financial metrics mentioned (e.g., Sharpe ratio, beta, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the response quality (0.0 to 1.0)")
    matlab_code: Optional[str] = Field(default=None, description="MATLAB code if applicable for calculations")
    follow_up_questions: List[str] = Field(default=[], description="Suggested follow-up questions for deeper understanding")
    risk_level: Optional[str] = Field(default=None, description="Risk level if applicable (low, medium, high)")
    time_horizon: Optional[str] = Field(default=None, description="Investment time horizon if applicable (short, medium, long)")
