"""
Master Execution Pipeline: Runs Experiments 001 through 004A sequentially.
"""

from experiments.exp001_momentum.run import run as run_001
from experiments.exp002_earnings_momentum.run import run as run_002
from experiments.exp003_complementarity.run import run as run_003
from experiments.exp004A_quality.run import run as run_004A
import reconcile

def main():
    print("=========================================================")
    print("STARTING COMPLETE RESEARCH PIPELINE: EXPERIMENTS 001 - 004A")
    print("=========================================================")
    
    run_001()
    run_002()
    run_003()
    run_004A()
    
    print("\n=========================================================")
    print("RUNNING AUTOMATED RECONCILIATION AGAINST LOCKED MANIFEST")
    print("=========================================================")
    reconcile.main()

if __name__ == "__main__":
    main()