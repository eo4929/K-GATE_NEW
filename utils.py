"""
K-GATE Utility Functions
Logging, retry logic, and helper functions
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import dotenv
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


# ===== Logging Setup =====
def setup_logging(output_dir: str = "results", verbose: bool = True) -> logging.Logger:
    """Setup logging configuration"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"kgate_{timestamp}.log"
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("KGATE")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


# ===== Environment Setup =====
def load_env(env_path: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file.
    Searches CWD first, then walks up to the project root."""
    # Try the given path first
    if not os.path.isfile(env_path):
        # Search upward from this file's directory
        search = Path(__file__).resolve().parent / ".env"
        if search.is_file():
            env_path = str(search)
    dotenv.load_dotenv(env_path)
    return dict(os.environ)


def get_openai_api_key(env_var: str = "OPENAI_API_KEY") -> str:
    """Get OpenAI API key from environment"""
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"OpenAI API key not found. "
            f"Set {env_var} environment variable or create .env file."
        )
    return api_key


# ===== Retry Decorator =====
def with_retry(max_attempts: int = 3, wait_min: float = 1.0, wait_max: float = 10.0):
    """Decorator for retry logic with exponential backoff"""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=wait_min, max=wait_max),
        reraise=True
    )


# ===== Output & Results =====
class ResultManager:
    """Manage experiment results and outputs"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.summary_file = self.output_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        self.results = []
        self.stats = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "avg_score": 0.0,
            "avg_num_nodes": 0.0,
            "node_count_distribution": {},
            "avg_frontier_size": 0.0,
            "branching_state_fraction": 0.0,
        }
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a single result"""
        self.results.append(result)
        
        # Update stats
        self.stats["total"] += 1
        if result.get("asr"):
            self.stats["successful"] += 1
        else:
            self.stats["failed"] += 1
        
        # Update average score
        scores = [r.get("score", 0) for r in self.results]
        self.stats["avg_score"] = sum(scores) / len(scores) if scores else 0.0

        # Structural DAG/MCTS diagnostics are computed from saved fields only;
        # no additional model calls are required.
        structural = [r for r in self.results if "num_nodes" in r]
        if structural:
            node_counts = [int(r["num_nodes"]) for r in structural]
            self.stats["avg_num_nodes"] = sum(node_counts) / len(node_counts)
            self.stats["node_count_distribution"] = {
                count: node_counts.count(count) for count in sorted(set(node_counts))
            }
            frontier_means = [
                float(r.get("mean_frontier_size", 0.0)) for r in structural
            ]
            branching_fractions = [
                float(r.get("branching_state_fraction", 0.0)) for r in structural
            ]
            self.stats["avg_frontier_size"] = (
                sum(frontier_means) / len(frontier_means)
            )
            self.stats["branching_state_fraction"] = (
                sum(branching_fractions) / len(branching_fractions)
            )
        
        # Append to JSONL file (streaming)
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def save_summary(self) -> None:
        """Save summary statistics"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": self.stats["total"],
            "successful": self.stats["successful"],
            "failed": self.stats["failed"],
            "asr": self.stats["successful"] / self.stats["total"] if self.stats["total"] > 0 else 0.0,
            "avg_harmfulness_score": self.stats["avg_score"],
            "avg_num_nodes": self.stats["avg_num_nodes"],
            "node_count_distribution": self.stats["node_count_distribution"],
            "avg_frontier_size": self.stats["avg_frontier_size"],
            "branching_state_fraction": self.stats["branching_state_fraction"],
            "results_file": str(self.results_file),
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def get_results(self) -> list:
        """Get all results"""
        return self.results
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return self.stats


# ===== Prompt Utils =====
class PromptLoader:
    """Load and format prompt templates from the prompts/ directory.

    Templates use Python str.format() placeholders (e.g., {title}, {description}).
    Files are read once and cached in memory.
    """

    _cache: Dict[str, str] = {}
    _prompts_dir: Optional[Path] = None

    @classmethod
    def _resolve_dir(cls) -> Path:
        if cls._prompts_dir is None:
            # Try relative to this file first, then CWD
            candidates = [
                Path(__file__).resolve().parent / "prompts",
                Path.cwd() / "prompts",
            ]
            for d in candidates:
                if d.is_dir():
                    cls._prompts_dir = d
                    break
            else:
                raise FileNotFoundError(
                    f"prompts/ directory not found. Searched: {[str(c) for c in candidates]}"
                )
        return cls._prompts_dir

    @classmethod
    def load_raw(cls, filename: str) -> str:
        """Load raw template string from prompts/<filename>."""
        if filename in cls._cache:
            return cls._cache[filename]
        filepath = cls._resolve_dir() / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt template not found: {filepath}")
        text = filepath.read_text(encoding="utf-8").strip()
        cls._cache[filename] = text
        return text

    @classmethod
    def load(cls, filename: str, **kwargs) -> str:
        """Load template and format with provided kwargs."""
        raw = cls.load_raw(filename)
        try:
            return raw.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing template variable {e} in {filename}. "
                f"Available keys: {list(kwargs.keys())}"
            )


class PromptTemplate:
    """Template utilities for prompts (legacy helper)"""

    @staticmethod
    def format_with_context(template: str, **kwargs) -> str:
        """Format template with context variables"""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    @staticmethod
    def truncate(text: str, max_length: int = 500) -> str:
        """Truncate text to max length"""
        if len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text


# ===== Validation Utils =====
class Validator:
    """Data validation utilities"""
    
    @staticmethod
    def is_valid_goal(goal: str, min_length: int = 5) -> bool:
        """Check if goal string is valid"""
        return isinstance(goal, str) and len(goal.strip()) >= min_length
    
    @staticmethod
    def is_valid_response(response: str, min_length: int = 10) -> bool:
        """Check if LLM response is valid"""
        return isinstance(response, str) and len(response.strip()) >= min_length
    
    @staticmethod
    def is_valid_score(score: int) -> bool:
        """Check if harmfulness score is valid"""
        return isinstance(score, int) and 0 <= score <= 10


# ===== Directory Utils =====
def ensure_dir(path: str) -> Path:
    """Ensure directory exists"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_csv_files(directory: str) -> list:
    """List all CSV files in directory"""
    return list(Path(directory).glob("*.csv"))


__all__ = [
    "setup_logging",
    "load_env",
    "get_openai_api_key",
    "with_retry",
    "ResultManager",
    "PromptLoader",
    "PromptTemplate",
    "Validator",
    "ensure_dir",
    "list_csv_files",
]
