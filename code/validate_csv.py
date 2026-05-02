import pandas as pd
import csv
import sys
import argparse
from pathlib import Path

def validate_and_clean_csv(file_path: str, output_path: str = None):
    print(f"Validating and cleaning: {file_path}")
    
    try:
        # Read the CSV (dtype=str prevents pandas from converting types automatically)
        df = pd.read_csv(file_path, dtype=str)
    except Exception as e:
        print(f"❌ Failed to parse CSV: {e}")
        sys.exit(1)
        
    # 1. Standardize empty values to empty string ("" not None)
    df = df.fillna("")
    df = df.replace("nan", "")
    df = df.replace("None", "")
    
    # 2. Remove newline breaks inside quoted fields
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: str(x).replace('\n', ' ').replace('\r', ' ').strip() 
            if isinstance(x, str) else x
        )
        
    # 3. Normalize request_type taxonomy
    valid_types = {"product_issue", "feature_request", "bug", "invalid"}
    if "request_type" in df.columns:
        df["request_type"] = df["request_type"].apply(
            lambda x: x.lower().strip() if x.lower().strip() in valid_types else "product_issue"
        )
        
    # 4. Strict CSV compliance (QUOTE_ALL)
    out_path = output_path if output_path else file_path
    try:
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Successfully wrote clean CSV to: {out_path}")
    except Exception as e:
        print(f"❌ Failed to write CSV: {e}")
        sys.exit(1)
        
    # 5. Final pandas validation pass
    try:
        df_validate = pd.read_csv(out_path)
        print(f"✅ Final Validation: {len(df_validate)} rows parsed cleanly in pandas.")
    except Exception as e:
        print(f"❌ Final Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and clean CSV output")
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to save cleaned CSV (defaults to overwriting input)", default=None)
    args = parser.parse_args()
    
    validate_and_clean_csv(args.input, args.output)
