"""Reuse the reviewed sealed-control identity with the unchanged full verifier."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.completion_development_verified_20261031 import study
from experiments.completion_verification_repair_20261031.verify_sealed import SealedVerifier
verifier=SealedVerifier(study)
evaluate,verify,summarize,run=verifier.evaluate,verifier.verify,verifier.summarize,verifier.run
if __name__=='__main__':run()
