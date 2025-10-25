#!/usr/bin/env python3
"""
AURELIA Lab 5 Evaluation System - Main Evaluation Runner

This script orchestrates the complete evaluation process:
1. Loads ground truth concepts from fintbx.pdf
2. Generates responses using the RAG system
3. Evaluates quality, latency, and cost metrics
4. Generates comprehensive reports
5. Optionally uploads results to GCS

Usage:
    python evaluate.py                           # Evaluate all concepts
    python evaluate.py --concepts "Duration" "Beta"  # Evaluate specific concepts
    python evaluate.py --force-refresh           # Force fresh generation (skip cache)
    python evaluate.py --no-gcs                  # Skip GCS upload
"""

import asyncio
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from backend.services.rag_service import rag_service
from backend.models.rag_models import RAGQuery, SearchStrategy
from evaluation_service import EvaluationService
from evaluation_models import (
    ConceptQualityMetrics, LatencyMetrics, TokenCostMetrics,
    EvaluationSummary, EvaluationResult
)


class EvaluationRunner:
    """Main evaluation orchestrator"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.evaluation_service = EvaluationService()
        self.evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = self.output_dir / self.evaluation_id
        
    async def run_evaluation(
        self,
        concepts: Optional[List[str]] = None,
        force_refresh: bool = False,
        upload_to_gcs: bool = True
    ) -> EvaluationSummary:
        """
        Run complete evaluation process
        
        Args:
            concepts: Specific concepts to evaluate (None = all)
            force_refresh: Force fresh generation (skip cache)
            upload_to_gcs: Upload results to Google Cloud Storage
            
        Returns:
            Complete evaluation summary
        """
        print(f"🚀 Starting AURELIA Lab 5 Evaluation")
        print(f"📁 Results will be saved to: {self.results_dir}")
        print(f"🆔 Evaluation ID: {self.evaluation_id}")
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ground truth data
        print("\n📚 Loading ground truth dataset...")
        ground_truth = await self._load_ground_truth()
        
        # Filter concepts if specified
        if concepts:
            ground_truth = {k: v for k, v in ground_truth.items() if k in concepts}
            print(f"🎯 Evaluating {len(ground_truth)} specified concepts: {list(ground_truth.keys())}")
        else:
            print(f"📊 Evaluating all {len(ground_truth)} concepts")
        
        # Initialize RAG service
        print("\n🔧 Initializing RAG service...")
        await rag_service.initialize_services()
        
        # Run evaluation for each concept
        print(f"\n🔄 Starting evaluation of {len(ground_truth)} concepts...")
        evaluation_results = []
        
        for i, (concept_name, ground_truth_data) in enumerate(ground_truth.items(), 1):
            print(f"\n📝 [{i}/{len(ground_truth)}] Evaluating: {concept_name}")
            
            try:
                result = await self._evaluate_concept(
                    concept_name, 
                    ground_truth_data, 
                    force_refresh
                )
                evaluation_results.append(result)
                
                # Print quick summary
                quality_score = result.quality_metrics.overall_score
                latency_ms = result.latency_metrics.total_time_ms
                print(f"   ✅ Quality: {quality_score:.1f}/100, Latency: {latency_ms:.0f}ms")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {concept_name}: {e}")
                # Create error result
                error_result = EvaluationResult(
                    concept_name=concept_name,
                    query=f"What is {concept_name}?",
                    generated_response="",
                    ground_truth=ground_truth_data,
                    quality_metrics=ConceptQualityMetrics(
                        accuracy_score=0.0,
                        completeness_score=0.0,
                        citation_fidelity_score=0.0,
                        overall_score=0.0
                    ),
                    latency_metrics=LatencyMetrics(
                        total_time_ms=0.0,
                        retrieval_time_ms=0.0,
                        generation_time_ms=0.0,
                        cache_hit=False
                    ),
                    token_cost_metrics=TokenCostMetrics(
                        total_tokens=0,
                        input_tokens=0,
                        output_tokens=0,
                        estimated_cost_usd=0.0
                    ),
                    error=str(e)
                )
                evaluation_results.append(error_result)
        
        # Generate summary
        print(f"\n📊 Generating evaluation summary...")
        summary = await self.evaluation_service.aggregate_results(evaluation_results)
        
        # Save results
        await self._save_results(evaluation_results, summary)
        
        # Upload to GCS if requested
        if upload_to_gcs:
            print(f"\n☁️  Uploading results to Google Cloud Storage...")
            await self._upload_to_gcs(summary)
        
        # Print final summary
        self._print_summary(summary)
        
        return summary
    
    async def _load_ground_truth(self) -> Dict[str, Any]:
        """Load ground truth concepts from JSON file"""
        ground_truth_path = Path(__file__).parent / "ground_truth.json"
        
        if not ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
        
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    
    async def _evaluate_concept(
        self, 
        concept_name: str, 
        ground_truth: Dict[str, Any], 
        force_refresh: bool
    ) -> EvaluationResult:
        """Evaluate a single concept"""
        
        # Generate query
        query = f"What is {concept_name}?"
        
        # Create RAG query
        rag_query = RAGQuery(
            query=query,
            strategy=SearchStrategy.RRF_FUSION,
            top_k=10,
            rerank_top_k=5,
            include_sources=True,
            enable_wikipedia_fallback=True,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Track timing
        start_time = time.time()
        
        # Generate response
        response = await rag_service.query(rag_query)
        
        total_time = (time.time() - start_time) * 1000
        
        # Check if response was cached
        cache_hit = response.metadata.get('cache_hit', False)
        
        # Evaluate quality
        quality_metrics = await self.evaluation_service.evaluate_quality(
            generated_response=response.answer,
            ground_truth=ground_truth,
            sources=response.sources
        )
        
        # Estimate token costs
        token_cost_metrics = await self.evaluation_service.estimate_token_cost(
            query=query,
            response=response.answer,
            sources=response.sources
        )
        
        # Create latency metrics
        latency_metrics = LatencyMetrics(
            total_time_ms=total_time,
            retrieval_time_ms=response.retrieval_time_ms or 0.0,
            generation_time_ms=response.generation_time_ms or 0.0,
            cache_hit=cache_hit
        )
        
        return EvaluationResult(
            concept_name=concept_name,
            query=query,
            generated_response=response.answer,
            ground_truth=ground_truth,
            quality_metrics=quality_metrics,
            latency_metrics=latency_metrics,
            token_cost_metrics=token_cost_metrics,
            sources=response.sources,
            metadata=response.metadata
        )
    
    async def _save_results(
        self, 
        evaluation_results: List[EvaluationResult], 
        summary: EvaluationSummary
    ):
        """Save evaluation results to local files"""
        
        # Save detailed results
        detailed_results = [result.dict() for result in evaluation_results]
        with open(self.results_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save summary
        with open(self.results_dir / "summary.json", 'w') as f:
            json.dump(summary.dict(), f, indent=2, default=str)
        
        # Generate markdown report
        report_content = await self.evaluation_service.generate_markdown_report(summary)
        with open(self.results_dir / "evaluation_report.md", 'w') as f:
            f.write(report_content)
        
        print(f"💾 Results saved to: {self.results_dir}")
    
    async def _upload_to_gcs(self, summary: EvaluationSummary):
        """Upload results to Google Cloud Storage"""
        try:
            # This would implement GCS upload functionality
            # For now, just log the intention
            print(f"☁️  Would upload to: gs://bucket/evaluations/{datetime.now().strftime('%Y-%m-%d')}/{self.evaluation_id}/")
            print(f"📊 Summary: {summary.overall_quality_score:.1f}/100 quality, {summary.average_latency_ms:.0f}ms latency")
        except Exception as e:
            print(f"⚠️  GCS upload failed: {e}")
    
    def _print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary to console"""
        print(f"\n{'='*60}")
        print(f"🎯 AURELIA Lab 5 Evaluation Complete")
        print(f"{'='*60}")
        print(f"📊 Overall Quality Score: {summary.overall_quality_score:.1f}/100")
        print(f"⚡ Average Latency: {summary.average_latency_ms:.0f}ms")
        print(f"💰 Total Estimated Cost: ${summary.total_cost_usd:.4f}")
        print(f"📈 Cache Hit Rate: {summary.cache_hit_rate:.1%}")
        print(f"🎯 Concepts Evaluated: {summary.total_concepts}")
        print(f"✅ Successful Evaluations: {summary.successful_evaluations}")
        print(f"❌ Failed Evaluations: {summary.failed_evaluations}")
        
        print(f"\n📋 Quality Breakdown:")
        print(f"   🎯 Accuracy: {summary.average_accuracy:.3f}")
        print(f"   📝 Completeness: {summary.average_completeness:.3f}")
        print(f"   📚 Citation Fidelity: {summary.average_citation_fidelity:.3f}")
        
        print(f"\n⚡ Performance Breakdown:")
        print(f"   🔍 Retrieval: {summary.average_retrieval_time_ms:.0f}ms")
        print(f"   🤖 Generation: {summary.average_generation_time_ms:.0f}ms")
        print(f"   💾 Cache Speedup: {summary.cache_speedup_factor:.1f}x")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AURELIA Lab 5 Evaluation System")
    parser.add_argument(
        "--concepts", 
        nargs="+", 
        help="Specific concepts to evaluate (default: all)"
    )
    parser.add_argument(
        "--force-refresh", 
        action="store_true", 
        help="Force fresh generation (skip cache)"
    )
    parser.add_argument(
        "--no-gcs", 
        action="store_true", 
        help="Skip Google Cloud Storage upload"
    )
    parser.add_argument(
        "--output-dir", 
        default="evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create evaluation runner
    runner = EvaluationRunner(output_dir=args.output_dir)
    
    try:
        # Run evaluation
        summary = await runner.run_evaluation(
            concepts=args.concepts,
            force_refresh=args.force_refresh,
            upload_to_gcs=not args.no_gcs
        )
        
        # Exit with appropriate code
        if summary.failed_evaluations > 0:
            print(f"\n⚠️  Evaluation completed with {summary.failed_evaluations} failures")
            sys.exit(1)
        else:
            print(f"\n✅ Evaluation completed successfully!")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
