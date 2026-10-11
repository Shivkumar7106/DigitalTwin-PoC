import os
import yaml
import logging
import random
from pathlib import Path
import numpy as np

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory.
    """
    # Assuming this file is located at src/utils/helpers.py
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str | Path) -> dict:
    """
    Loads a YAML configuration file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        Dictionary containing the configuration.
    """
    path = Path(config_path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dir(path: str | Path) -> None:
    """
    Creates a directory if it does not exist.
    
    Args:
        path: Path to the directory.
    """
    os.makedirs(path, exist_ok=True)

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures standard logging for the project.
    
    Args:
        level: Logging level (e.g., logging.INFO).
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def set_random_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for reproducibility across random, numpy, and tensorflow.
    
    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        logging.warning("TensorFlow is not installed. Skipping TF seed setting.")
