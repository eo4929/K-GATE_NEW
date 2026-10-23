"""
K-GATE Data Loader Module
Load and preprocess AdvBench and HarmBench datasets
"""

import os
import csv
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from config import DatasetConfig


class DataLoader:
    """AdvBench 및 HarmBench CSV 파일 로드 및 전처리"""
    
    def __init__(self, data_dir: str = "data"):
        if data_dir == "data":
            current_path = Path(__file__).resolve().parent
            self.data_dir = None
            # Search upward for the 'data' directory starting from this file's location.
            for parent in [current_path] + list(current_path.parents):
                candidate = parent / "data"
                if candidate.exists():
                    self.data_dir = candidate
                    break
            if self.data_dir is None:
                self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_advbench(self, filepath: Optional[str] = None, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load AdvBench dataset
        
        Expected CSV format:
        - Columns: goal (or behavior), category
        
        Args:
            filepath: Custom filepath (default: data/advbench_behaviors.csv)
            num_samples: Limit number of samples (None = all)
            
        Returns:
            List of dicts with 'goal' and optional 'category' keys
        """
        if filepath is None:
            filepath = self.data_dir / "advbench_behaviors.csv"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        goals = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV file: {filepath}")
            
            # Determine goal column name
            goal_col = None
            for col_name in ['goal', 'behavior', 'Goal', 'Behavior']:
                if col_name in reader.fieldnames:
                    goal_col = col_name
                    break
            
            if goal_col is None:
                raise ValueError(f"No 'goal' or 'behavior' column found. Columns: {reader.fieldnames}")
            
            # Find category column if exists
            category_col = None
            for col_name in ['category', 'Category', 'cat']:
                if col_name in reader.fieldnames:
                    category_col = col_name
                    break
            
            for row in reader:
                goal = row.get(goal_col, "").strip()
                if goal:
                    entry = {
                        "goal": goal,
                        "dataset": "advbench",
                    }
                    if category_col:
                        entry["category"] = row.get(category_col, "unknown").strip()
                    goals.append(entry)
        
        if num_samples:
            random.seed(42)
            goals = random.sample(goals, min(num_samples, len(goals)))
        
        return goals
    
    def load_harmbench(self, filepath: Optional[str] = None, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load HarmBench dataset
        
        Expected CSV format:
        - Columns: goal (or text), category, risk levels
        
        Args:
            filepath: Custom filepath (default: data/harmbench_behaviors_text_all.csv)
            num_samples: Limit number of samples (None = all)
            
        Returns:
            List of dicts with 'goal' and 'category' keys
        """
        if filepath is None:
            filepath = self.data_dir / "harmbench_behaviors_text_all.csv"
        else:
            filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        goals = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV file: {filepath}")
            
            # Determine goal column name
            goal_col = None
            for col_name in ['goal', 'text', 'Goal', 'Text', 'behavior', 'Behavior']:
                if col_name in reader.fieldnames:
                    goal_col = col_name
                    break
            
            if goal_col is None:
                raise ValueError(f"No goal/text column found. Columns: {reader.fieldnames}")
            
            # Find category column if exists
            category_col = None
            for col_name in ['category', 'Category', 'cat', 'harm_category', 'SemanticCategory', 'FunctionalCategory']:
                if col_name in reader.fieldnames:
                    category_col = col_name
                    break
            
            for row in reader:
                goal = row.get(goal_col, "").strip()
                if goal:
                    entry = {
                        "goal": goal,
                        "dataset": "harmbench",
                    }
                    if category_col:
                        entry["category"] = row.get(category_col, "unknown").strip()
                    goals.append(entry)
        
        if num_samples:
            random.seed(42)
            goals = random.sample(goals, min(num_samples, len(goals)))
        
        return goals
    
    def load_dataset(self, dataset_name: str, num_samples: Optional[int] = None) -> List[Dict]:
        """
        Load dataset by name
        
        Args:
            dataset_name: 'advbench' or 'harmbench'
            num_samples: Limit number of samples
            
        Returns:
            List of goal dictionaries
        """
        if dataset_name.lower() == "advbench":
            return self.load_advbench(num_samples=num_samples)
        elif dataset_name.lower() == "harmbench":
            return self.load_harmbench(num_samples=num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def load_all(self, num_samples_per_dataset: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Load both AdvBench and HarmBench datasets
        
        Returns:
            Tuple of (advbench_goals, harmbench_goals)
        """
        advbench = self.load_advbench(num_samples=num_samples_per_dataset)
        harmbench = self.load_harmbench(num_samples=num_samples_per_dataset)
        return advbench, harmbench


class DatasetValidator:
    """데이터셋 유효성 검증"""
    
    @staticmethod
    def validate_goals(goals: List[Dict]) -> bool:
        """Validate that all goals have required fields"""
        if not goals:
            return False
        
        for i, goal in enumerate(goals):
            if "goal" not in goal or not goal["goal"]:
                raise ValueError(f"Goal {i} has missing or empty 'goal' field: {goal}")
            if not isinstance(goal["goal"], str):
                raise ValueError(f"Goal {i} 'goal' field is not string: {goal}")
        
        return True
    
    @staticmethod
    def filter_goals(goals: List[Dict], min_length: int = 10, max_length: int = 1000) -> List[Dict]:
        """Filter goals by length"""
        filtered = []
        for goal in goals:
            length = len(goal["goal"])
            if min_length <= length <= max_length:
                filtered.append(goal)
        return filtered


__all__ = [
    "DataLoader",
    "DatasetValidator",
]
