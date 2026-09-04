"""
Automated Reconciliation Engine.
Compares actual output metrics against config/locked_results.yaml tolerances.
"""

import yaml
import pandas as pd
import numpy as np
from src.utils.io import load_csv, get_path, save_csv

def main():
    with open(get_path("config/locked_results.yaml"), "r") as f:
        locked = yaml.safe_load(f)
        
    e001_df = load_csv("outputs/tables/exp_001_ic_summary.csv")
    e002_df = load_csv("outputs/tables/exp_002_ic_summary.csv")
    e003_df = load_csv("outputs/tables/exp_003_summary.csv")
    e004_df = load_csv("outputs/tables/exp_004A_paired_delta_ic.csv")
    
    reconciliation = []
    
    # 001 Checks
    sec_60_row = e001_df[e001_df['factor'] == 'sector_relative_mom_60d'].iloc[0]
    exp = locked['experiment_001']['sector_relative_60d_ic']
    act = sec_60_row['research_ic']
    diff = act - exp['expected']
    passed = abs(diff) <= exp['tolerance']
    reconciliation.append({
        'experiment': '001', 'metric': 'sector_relative_60d_ic',
        'expected': exp['expected'], 'actual': act, 'difference': diff,
        'tolerance': exp['tolerance'], 'status': 'PASS' if passed else 'FAIL'
    })
    
    # 002 Checks
    eps_acc_row = e002_df[e002_df['factor'] == 'eps_growth_acceleration'].iloc[0]
    exp = locked['experiment_002']['eps_accel_ic']
    act = eps_acc_row['research_ic']
    diff = act - exp['expected']
    passed = abs(diff) <= exp['tolerance']
    reconciliation.append({
        'experiment': '002', 'metric': 'eps_accel_ic',
        'expected': exp['expected'], 'actual': act, 'difference': diff,
        'tolerance': exp['tolerance'], 'status': 'PASS' if passed else 'FAIL'
    })
    
    # 003 Checks
    base_ic_val = e003_df[e003_df['metric'] == 'BaseScore_IC_Research']['value'].values[0]
    exp = locked['experiment_003']['basescore_ic']
    diff = base_ic_val - exp['expected']
    passed = abs(diff) <= exp['tolerance']
    reconciliation.append({
        'experiment': '003', 'metric': 'basescore_ic',
        'expected': exp['expected'], 'actual': base_ic_val, 'difference': diff,
        'tolerance': exp['tolerance'], 'status': 'PASS' if passed else 'FAIL'
    })
    
    # 004A Checks
    roe_delta_val = e004_df[e004_df['comparison'].str.contains('ROE')]['mean_delta_ic'].values[0]
    exp = locked['experiment_004A']['roe_delta_ic']
    diff = roe_delta_val - exp['expected']
    passed = abs(diff) <= exp['tolerance']
    reconciliation.append({
        'experiment': '004A', 'metric': 'roe_delta_ic',
        'expected': exp['expected'], 'actual': roe_delta_val, 'difference': diff,
        'tolerance': exp['tolerance'], 'status': 'PASS' if passed else 'FAIL'
    })
    
    rec_df = pd.DataFrame(reconciliation)
    save_csv(rec_df, "outputs/reports/reconciliation_report.csv")
    
    print(rec_df.to_string(index=False))
    
    if (rec_df['status'] == 'FAIL').any():
        print("\nRECONCILIATION FAILED! At least one metric exceeded tolerance threshold.")
        sys.exit(1)
    else:
        print("\nRECONCILIATION SUCCESSFUL! All metrics within locked tolerances.")

if __name__ == "__main__":
    import sys
    main()