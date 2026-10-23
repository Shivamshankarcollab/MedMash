import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_phase1():
    print("[1/4] Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    
    total_samples = len(df)
    print(f"Total dataset records: {total_samples}")

    # 1. Stratified split across 3 hospitals (230, 190, 149 = 569)
    df_aiims, df_remaining = train_test_split(
        df, 
        train_size=230, 
        stratify=df['target'], 
        random_state=42
    )
    
    df_tata, df_apollo = train_test_split(
        df_remaining, 
        train_size=190, 
        stratify=df_remaining['target'], 
        random_state=42
    )

    silos = {
        "aiims": df_aiims,
        "tata": df_tata,
        "apollo": df_apollo
    }

    base_dir = "data"
    os.makedirs(base_dir, exist_ok=True)

    print("\n[2/4] Processing silos: Splitting & Local Scaling...")
    for hospital_name, hosp_df in silos.items():
        hosp_dir = os.path.join(base_dir, hospital_name)
        os.makedirs(hosp_dir, exist_ok=True)

        # 80/20 train/test split
        local_train, local_test = train_test_split(
            hosp_df, 
            test_size=0.20, 
            stratify=hosp_df['target'], 
            random_state=42
        )

        X_train = local_train.drop(columns=['target'])
        y_train = local_train['target'].values
        X_test = local_test.drop(columns=['target'])
        y_test = local_test['target'].values

        # Local scaling rule: Fit ONLY on train, transform on test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save Scaler for inference
        joblib.dump(scaler, os.path.join(hosp_dir, "scaler.pkl"))

        # Combine and save CSVs
        train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        train_final['target'] = y_train
        train_final.to_csv(os.path.join(hosp_dir, "train.csv"), index=False)

        test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        test_final['target'] = y_test
        test_final.to_csv(os.path.join(hosp_dir, "test.csv"), index=False)

        print(f"  [{hospital_name.upper()}] Done -> Train: {len(train_final)}, Test: {len(test_final)}")

    print("\n[3/4] All silos partitioned with zero leakage.")

if __name__ == "__main__":
    run_phase1()
