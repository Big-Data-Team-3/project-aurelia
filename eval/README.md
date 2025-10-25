# AURELIA Lab 5 Evaluation System

A comprehensive evaluation framework for the AURELIA RAG system, designed to assess quality, performance, and cost metrics across financial concepts.

## Overview

The AURELIA Lab 5 Evaluation System provides:

- **Quality Assessment**: Accuracy, completeness, and citation fidelity metrics
- **Performance Analysis**: Latency, caching effectiveness, and speedup measurements  
- **Cost Estimation**: Token usage and API cost tracking
- **Comprehensive Reporting**: Detailed markdown reports and data exports
- **Remote Testing**: Production API evaluation capabilities

## Quick Start

### 1. Local Evaluation

Evaluate all concepts using the local RAG system:

```bash
cd eval
python evaluate.py
```

### 2. Remote Evaluation

Test deployed API endpoints:

```bash
cd eval
python evaluate_remote.py --api-url https://your-api.com
```

### 3. Specific Concepts

Evaluate only specific financial concepts:

```bash
python evaluate.py --concepts "Duration" "Beta" "Sharpe Ratio"
```

## Features

### Quality Metrics

- **Accuracy (40% weight)**: Semantic similarity between generated and ground truth definitions
- **Completeness (30% weight)**: Field coverage including definition, components, examples, use cases, and formulas
- **Citation Fidelity (30% weight)**: Page reference accuracy using F1 score

### Performance Metrics

- **Latency Analysis**: Total, retrieval, and generation times
- **Cache Effectiveness**: Hit rates and speedup factors
- **Component Breakdown**: Detailed timing for each pipeline stage

### Cost Tracking

- **Token Usage**: Input, output, and total token counts
- **Cost Estimation**: Embedding and generation API costs
- **Per-Query Analysis**: Average cost per evaluation

## Ground Truth Dataset

The evaluation uses 15 financial concepts extracted from `fintbx.pdf`:

1. **Duration** - Bond price sensitivity to interest rates
2. **Beta** - Security volatility relative to market
3. **Sharpe Ratio** - Risk-adjusted performance measure
4. **Value at Risk (VaR)** - Maximum potential loss measure
5. **Monte Carlo Simulation** - Random sampling computational method
6. **Black-Scholes Model** - European option pricing model
7. **Capital Asset Pricing Model (CAPM)** - Risk-return relationship
8. **Modern Portfolio Theory (MPT)** - Diversification framework
9. **Arbitrage Pricing Theory (APT)** - Multi-factor risk model
10. **Greeks (Options)** - Option sensitivity measures
11. **Credit Risk** - Borrower default risk
12. **Liquidity Risk** - Asset conversion risk
13. **Stress Testing** - Extreme scenario evaluation
14. **Risk-Adjusted Return** - Performance normalization
15. **Derivatives Pricing** - Financial derivative valuation

Each concept includes:
- Complete definition and explanation
- Key components and characteristics
- Relevant formulas and calculations
- Practical examples and use cases
- Page references for citation validation
- Confidence scores for quality assessment

## Usage Examples

### Basic Evaluation

```bash
# Evaluate all concepts
python evaluate.py

# Evaluate specific concepts
python evaluate.py --concepts "Duration" "Beta"

# Force fresh generation (skip cache)
python evaluate.py --force-refresh

# Skip GCS upload
python evaluate.py --no-gcs
```

### Remote API Testing

```bash
# Test local API
python evaluate_remote.py

# Test staging API
python evaluate_remote.py --api-url https://staging-api.aurelia.com

# Test production API
python evaluate_remote.py --api-url https://api.aurelia.com

# Disable caching
python evaluate_remote.py --no-cache
```

### Advanced Configuration

```bash
# Custom output directory
python evaluate.py --output-dir custom_results

# Custom API timeout
EVAL_TIMEOUT_SECONDS=120 python evaluate_remote.py

# Enable detailed logging
EVAL_LOG_LEVEL=DEBUG python evaluate.py
```

## Output Structure

Each evaluation generates:

```
evaluation_results/
└── eval_YYYYMMDD_HHMMSS/
    ├── summary.json              # High-level metrics
    ├── detailed_results.json     # Per-concept details
    ├── evaluation_report.md      # Human-readable report
    ├── api_config.json          # API configuration (remote only)
    └── evaluation.log           # Detailed logs
```

### Summary Metrics

- **Overall Quality Score**: Weighted average (0-100)
- **Success Rate**: Percentage of successful evaluations
- **Average Latency**: Mean response time across all queries
- **Cache Hit Rate**: Percentage of cached responses
- **Total Cost**: Estimated API costs for evaluation

### Quality Distribution

- **High Quality (>80)**: Excellent responses with comprehensive information
- **Medium Quality (60-80)**: Good responses with minor gaps
- **Low Quality (<60)**: Responses requiring improvement

### Performance Distribution

- **Fast Responses (<1000ms)**: Cached or very efficient queries
- **Medium Responses (1000-3000ms)**: Typical fresh query performance
- **Slow Responses (>3000ms)**: Queries requiring optimization

## Configuration

### Environment Variables

```bash
# API Configuration
EVAL_API_URL=https://api.aurelia.com
EVAL_TIMEOUT_SECONDS=60
EVAL_MAX_RETRIES=3

# Quality Thresholds
EVAL_HIGH_QUALITY_THRESHOLD=80
EVAL_MEDIUM_QUALITY_THRESHOLD=60

# Performance Thresholds
EVAL_FAST_RESPONSE_THRESHOLD_MS=1000
EVAL_SLOW_RESPONSE_THRESHOLD_MS=3000

# Cost Configuration
EVAL_EMBEDDING_COST_PER_1K_TOKENS=0.13
EVAL_GENERATION_INPUT_COST_PER_1K_TOKENS=0.15
EVAL_GENERATION_OUTPUT_COST_PER_1K_TOKENS=0.60

# Features
EVAL_ENABLE_CACHING=true
EVAL_FORCE_REFRESH=false
EVAL_UPLOAD_TO_GCS=true
```

### Quality Weights

The evaluation uses weighted scoring:

- **Accuracy**: 40% - Semantic similarity to ground truth
- **Completeness**: 30% - Field coverage and detail
- **Citation Fidelity**: 30% - Page reference accuracy

## Interpretation Guide

### Quality Scores

- **90-100**: Excellent - Comprehensive, accurate, well-cited
- **80-89**: Good - Mostly complete with minor gaps
- **70-79**: Fair - Adequate but missing some elements
- **60-69**: Poor - Significant gaps or inaccuracies
- **<60**: Failed - Major issues requiring attention

### Performance Benchmarks

- **<500ms**: Excellent - Cached responses or very efficient
- **500-1000ms**: Good - Fast fresh responses
- **1000-2000ms**: Acceptable - Typical fresh query performance
- **2000-3000ms**: Slow - May need optimization
- **>3000ms**: Poor - Requires investigation

### Cost Guidelines

- **<$0.01**: Very efficient - Minimal token usage
- **$0.01-0.05**: Efficient - Reasonable cost per query
- **$0.05-0.10**: Moderate - Higher complexity queries
- **>$0.10**: Expensive - May need optimization

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check API URL and network connectivity
   - Verify API is running and accessible
   - Check authentication if required

2. **Evaluation Timeouts**
   - Increase timeout settings
   - Check API performance
   - Consider reducing batch size

3. **Low Quality Scores**
   - Review ground truth data
   - Check RAG system configuration
   - Verify document retrieval quality

4. **High Latency**
   - Enable caching for repeated queries
   - Check API performance
   - Optimize retrieval strategies

### Debug Mode

Enable detailed logging:

```bash
EVAL_LOG_LEVEL=DEBUG python evaluate.py
```

## Integration

### CI/CD Pipeline

Add evaluation to your deployment pipeline:

```yaml
- name: Run RAG Evaluation
  run: |
    cd eval
    python evaluate_remote.py --api-url ${{ env.API_URL }}
```

### Monitoring

Set up automated evaluation runs:

```bash
# Daily evaluation
0 2 * * * cd /path/to/eval && python evaluate_remote.py --api-url https://api.aurelia.com
```

## Contributing

To add new evaluation concepts:

1. Add concept to `ground_truth.json`
2. Include definition, components, examples, use cases
3. Add page references for citation validation
4. Set appropriate confidence score

## Support

For issues or questions:

- Check logs in `evaluation_results/eval_*/evaluation.log`
- Review configuration in `eval_config.py`
- Verify API connectivity and health
- Consult ground truth data accuracy

---

*AURELIA Lab 5 Evaluation System v1.0.0*
