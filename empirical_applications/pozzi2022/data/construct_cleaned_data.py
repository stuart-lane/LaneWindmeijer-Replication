import pandas as pd
import numpy as np
import os

# Get directory, input file and output file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Countries to include
COUNTRIES = ['AUS', 'DNK', 'FIN', 'FRA', 'ITA', 'NLD', 'NOR', 'PRT', 'ESP', 'SWE', 'CHE', 'GBR', 'USA']

def main():    
    INPUT_FILE = 'JSTdatasetR6.xlsx'
    OUTPUT_FILE = 'data.txt'

    # Make paths absolute if they're relative
    if not os.path.isabs(INPUT_FILE):
        INPUT_FILE = os.path.join(SCRIPT_DIR, INPUT_FILE)
    if not os.path.isabs(OUTPUT_FILE):
        OUTPUT_FILE = os.path.join(SCRIPT_DIR, OUTPUT_FILE)
    
    # Load data
    df = pd.read_excel(INPUT_FILE)
    
    # Filter to relevant countries and years (1949-2015; need 1949 for 1950 growth rates)
    df = df[df['iso'].isin(COUNTRIES)].copy()
    df = df[(df['year'] >= 1949) & (df['year'] <= 2015)].copy()
    df = df.sort_values(['country', 'year']).reset_index(drop=True)
    
    # Calculate inflation from CPI
    df['inflation'] = df.groupby('country')['cpi'].pct_change()
    
    # Calculate real returns
    df['r_eq_tr'] = (1 + df['eq_tr']) / (1 + df['inflation']) - 1
    df['r_housing_tr'] = (1 + df['housing_tr']) / (1 + df['inflation']) - 1
    
    # Calculate consumption growth (log difference)
    df['dc'] = df.groupby('country')['rconsbarro'].apply(lambda x: np.log(x).diff()).reset_index(level=0, drop=True)
    
    # Select final columns and filter to 1950-2015
    final_columns = ['iso', 'country', 'year', 'eq_tr', 'r_eq_tr', 'housing_tr', 'r_housing_tr', 'dc']
    df_final = df[df['year'] >= 1950][final_columns].copy()
    
    # Save
    df_final.to_csv(OUTPUT_FILE, sep='\t', index=False, float_format='%.5f')
    print(f"Saved {len(df_final)} rows to {OUTPUT_FILE}")
    
    return df_final

if __name__ == "__main__":
    main()