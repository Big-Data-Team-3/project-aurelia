#!/usr/bin/env python3
"""
AURELIA Lab 5 Evaluation System - Test Script
Simple test to verify evaluation system functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from evaluation_service import EvaluationService
from evaluation_models import ConceptQualityMetrics, LatencyMetrics, TokenCostMetrics


async def test_evaluation_service():
    """Test basic evaluation service functionality"""
    print("🧪 Testing AURELIA Lab 5 Evaluation Service...")
    
    # Initialize evaluation service
    eval_service = EvaluationService()
    
    # Test data
    test_ground_truth = {
        "name": "Duration",
        "definition": "Duration is a measure of the sensitivity of a bond's price to changes in interest rates.",
        "key_components": ["Weighted average time", "Present value", "Interest rate sensitivity"],
        "formulas": ["Duration = Σ(t × PV(CF_t)) / Σ(PV(CF_t))"],
        "examples": ["A 5-year bond has duration of 4.2 years"],
        "use_cases": ["Interest rate risk management", "Portfolio immunization"],
        "page_references": [45, 46, 47],
        "confidence_score": 0.95
    }
    
    test_response = """
    Duration is a financial measure that quantifies how sensitive a bond's price is to changes in interest rates. 
    It represents the weighted average time until a bond's cash flows are received.
    
    Key components include:
    1. Weighted average time to maturity
    2. Present value of cash flows  
    3. Interest rate sensitivity
    
    Formula: Duration = Σ(t × PV(CF_t)) / Σ(PV(CF_t))
    
    For example, a 5-year bond with annual coupons typically has a duration of approximately 4.2 years.
    
    Use cases include interest rate risk management and portfolio immunization strategies.
    """
    
    test_sources = [
        {"text": "Duration measures bond price sensitivity...", "pages": [45, 46]},
        {"text": "The formula for duration is...", "pages": [47]}
    ]
    
    try:
        # Test quality evaluation
        print("📊 Testing quality evaluation...")
        quality_metrics = await eval_service.evaluate_quality(
            generated_response=test_response,
            ground_truth=test_ground_truth,
            sources=test_sources
        )
        
        print(f"   ✅ Quality Score: {quality_metrics.overall_score:.1f}/100")
        print(f"   ✅ Accuracy: {quality_metrics.accuracy_score:.3f}")
        print(f"   ✅ Completeness: {quality_metrics.completeness_score:.3f}")
        print(f"   ✅ Citation Fidelity: {quality_metrics.citation_fidelity_score:.3f}")
        
        # Test cost estimation
        print("💰 Testing cost estimation...")
        cost_metrics = await eval_service.estimate_token_cost(
            query="What is Duration?",
            response=test_response,
            sources=test_sources
        )
        
        print(f"   ✅ Total Tokens: {cost_metrics.total_tokens}")
        print(f"   ✅ Estimated Cost: ${cost_metrics.total_cost_usd:.4f}")
        
        # Test report generation
        print("📝 Testing report generation...")
        from evaluation_models import EvaluationSummary
        
        test_summary = EvaluationSummary(
            evaluation_id="test_eval",
            total_concepts=1,
            successful_evaluations=1,
            failed_evaluations=0,
            overall_quality_score=quality_metrics.overall_score,
            average_accuracy=quality_metrics.accuracy_score,
            average_completeness=quality_metrics.completeness_score,
            average_citation_fidelity=quality_metrics.citation_fidelity_score,
            average_latency_ms=1500.0,
            average_retrieval_time_ms=200.0,
            average_generation_time_ms=1200.0,
            cache_hit_rate=0.0,
            cache_speedup_factor=1.0,
            total_cost_usd=cost_metrics.total_cost_usd,
            average_cost_per_query_usd=cost_metrics.total_cost_usd,
            total_tokens_used=cost_metrics.total_tokens,
            duration_minutes=1.0
        )
        
        report = await eval_service.generate_markdown_report(test_summary)
        print(f"   ✅ Report generated ({len(report)} characters)")
        
        print("\n✅ All tests passed! Evaluation system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_ground_truth_loading():
    """Test ground truth data loading"""
    print("\n📚 Testing ground truth data loading...")
    
    try:
        import json
        ground_truth_path = Path(__file__).parent / "ground_truth.json"
        
        if not ground_truth_path.exists():
            print("   ❌ Ground truth file not found")
            return False
        
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        
        print(f"   ✅ Loaded {len(ground_truth)} concepts")
        
        # Check a few concepts
        required_concepts = ["Duration", "Beta", "Sharpe Ratio"]
        for concept in required_concepts:
            if concept in ground_truth:
                data = ground_truth[concept]
                print(f"   ✅ {concept}: {data.get('definition', 'No definition')[:50]}...")
            else:
                print(f"   ❌ Missing concept: {concept}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Ground truth loading failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🚀 AURELIA Lab 5 Evaluation System - Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Ground truth loading
    if await test_ground_truth_loading():
        tests_passed += 1
    
    # Test 2: Evaluation service
    if await test_evaluation_service():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Evaluation system is ready to use.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
