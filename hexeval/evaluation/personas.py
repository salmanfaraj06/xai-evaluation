"""
Utilities for loading persona definitions from configuration.
"""
from pathlib import Path
from typing import List, Dict
import yaml
import logging

LOG = logging.getLogger(__name__)

def load_personas_from_file(path: str | Path) -> List[Dict]:
    """
    Load personas from a YAML file.
    
    Parameters
    ----------
    path : str or Path
        Path to the YAML file containing persona definitions.
        
    Returns
    -------
    List[Dict]
        List of persona dictionaries.
    """
    path = Path(path)
    if not path.exists():
       
        project_root = Path(__file__).parent.parent.parent
        possible_path = project_root / path
        if possible_path.exists():
            path = possible_path
        else:
            raise FileNotFoundError(f"Personas file not found at {path} or {possible_path}")
            
    with open(path, "r") as f:
        personas = yaml.safe_load(f)
        
    if not isinstance(personas, list):
        raise ValueError(f"Personas file must contain a list of persona dictionaries, got {type(personas)}")
        
    LOG.info(f"Loaded {len(personas)} personas from {path.name}")
    return personas
