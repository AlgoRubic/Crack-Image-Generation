"""Forward to the canonical evaluator to prevent copied-metric drift."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'src' / 'evaluate_exp6_spatial_physics_256.py'), run_name='__main__')
