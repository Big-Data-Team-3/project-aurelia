"""
AURELIA Lab 5 Evaluation Configuration
Centralized configuration for evaluation runs
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from evaluation_models import EvaluationConfig


class EvaluationSettings:
    """Evaluation settings and configuration management"""
    
    # Default configuration
    DEFAULT_CONFIG = EvaluationConfig()
    
    # Environment-specific configurations
    ENVIRONMENTS = {
        "local": {
            "api_url": "http://localhost:8000",
            "timeout_seconds": 60,
            "max_retries": 3,
            "enable_caching": True
        },
        "staging": {
            "api_url": "https://staging-api.aurelia.com",
            "timeout_seconds": 90,
            "max_retries": 5,
            "enable_caching": True
        },
        "production": {
            "api_url": "https://api.aurelia.com",
            "timeout_seconds": 120,
            "max_retries": 5,
            "enable_caching": True
        }
    }
    
    @classmethod
    def get_config(cls, environment: str = "local") -> EvaluationConfig:
        """Get evaluation configuration for specific environment"""
        env_config = cls.ENVIRONMENTS.get(environment, cls.ENVIRONMENTS["local"])
        
        # Override with environment variables if available
        config_dict = cls.DEFAULT_CONFIG.dict()
        config_dict.update(env_config)
        
        # Apply environment variable overrides
        for key, value in os.environ.items():
            if key.startswith("EVAL_"):
                config_key = key[5:].lower()
                if config_key in config_dict:
                    # Convert string values to appropriate types
                    if isinstance(config_dict[config_key], bool):
                        config_dict[config_key] = value.lower() in ("true", "1", "yes")
                    elif isinstance(config_dict[config_key], int):
                        config_dict[config_key] = int(value)
                    elif isinstance(config_dict[config_key], float):
                        config_dict[config_key] = float(value)
                    else:
                        config_dict[config_key] = value
        
        return EvaluationConfig(**config_dict)
    
    @classmethod
    def get_api_url(cls, environment: str = "local") -> str:
        """Get API URL for specific environment"""
        return cls.ENVIRONMENTS.get(environment, cls.ENVIRONMENTS["local"])["api_url"]
    
    @classmethod
    def get_timeout(cls, environment: str = "local") -> int:
        """Get timeout for specific environment"""
        return cls.ENVIRONMENTS.get(environment, cls.ENVIRONMENTS["local"])["timeout_seconds"]


class EvaluationPaths:
    """Path management for evaluation files"""
    
    def __init__(self, base_dir: str = "evaluation_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def get_results_dir(self, evaluation_id: str) -> Path:
        """Get results directory for specific evaluation"""
        return self.base_dir / evaluation_id
    
    def get_ground_truth_path(self) -> Path:
        """Get path to ground truth file"""
        return Path(__file__).parent / "ground_truth.json"
    
    def get_config_path(self) -> Path:
        """Get path to evaluation configuration"""
        return Path(__file__).parent / "eval_config.json"
    
    def get_log_path(self, evaluation_id: str) -> Path:
        """Get log file path for evaluation"""
        return self.base_dir / evaluation_id / "evaluation.log"


class EvaluationLogger:
    """Logging configuration for evaluations"""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_evaluation_start(self, evaluation_id: str, concepts: list):
        """Log evaluation start"""
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Evaluation Started: {evaluation_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Concepts: {', '.join(concepts)}\n")
            f.write(f"{'='*60}\n")
    
    def log_concept_result(self, concept: str, quality_score: float, latency_ms: float, error: str = None):
        """Log individual concept result"""
        with open(self.log_path, 'a') as f:
            timestamp = datetime.now().strftime('%H:%M:%S')
            if error:
                f.write(f"[{timestamp}] {concept}: ERROR - {error}\n")
            else:
                f.write(f"[{timestamp}] {concept}: Quality={quality_score:.1f}, Latency={latency_ms:.0f}ms\n")
    
    def log_evaluation_end(self, evaluation_id: str, summary: Dict[str, Any]):
        """Log evaluation completion"""
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Evaluation Completed: {evaluation_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Overall Quality: {summary.get('overall_quality_score', 0):.1f}/100\n")
            f.write(f"Average Latency: {summary.get('average_latency_ms', 0):.0f}ms\n")
            f.write(f"Success Rate: {summary.get('successful_evaluations', 0)}/{summary.get('total_concepts', 0)}\n")
            f.write(f"{'='*60}\n")


# Global configuration instance
eval_config = EvaluationSettings.get_config()
eval_paths = EvaluationPaths()
