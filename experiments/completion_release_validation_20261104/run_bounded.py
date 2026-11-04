"""One original 2880-second release-validation budget; no automatic retries."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.completion_release_validation_20261104 import study
from experiments.compositional_completion_20261031.harness import bounded


def main():return bounded(study)
if __name__=='__main__':main()
