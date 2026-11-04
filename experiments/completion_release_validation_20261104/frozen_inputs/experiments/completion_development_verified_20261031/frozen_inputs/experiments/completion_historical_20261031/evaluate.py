"""Score sealed models and retain complete-denominator failure receipts."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.completion_historical_20261031 import study
from experiments.compositional_completion_20261031.verification import Verifier
verifier=Verifier(study)
evaluate,verify,summarize,run=verifier.evaluate,verifier.verify,verifier.summarize,verifier.run
if __name__=='__main__':run()
