import os
import numpy as np
import pandas as pd

from score_test import ScoreTest
from application_utils import *
from config import *

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

original_df = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE_POZZI))

df, table_header, countries = clean_pozzi_data(original_df)

tables = [1, 2]

def main():

    for table in tables:

        results_list = []

        lagt = table

        for country in countries:
                
            df_country = df[df['iso'] == country].copy()
            df_check = df_country
            df_country = df_country.dropna()

            df_country['mean_rhr'] = np.nan

            for idx, row in df_country.iterrows():
                year = row['year']

                other_countries_values = []
                for other_country in countries:
                    if other_country != country:
                        other_df = df[df['iso'] == other_country].copy()
                        other_df = other_df.dropna()
                        year_match = other_df[other_df['year'] == year]['r_housing_tr']
                        if not year_match.empty:
                            other_countries_values.append(year_match.values[0])
                
                if other_countries_values:
                    df_country.at[idx, 'mean_rhr'] = np.mean(other_countries_values)

            rhr = df_country['r_housing_tr'].values
            dc = df_country['dc'].values
            averagerate = df_country['mean_rhr'].values

            y = dc[2:].reshape(-1, 1)             
            X = rhr[2:].reshape(-1, 1)               
            Z1 = rhr[(2-lagt):-lagt].reshape(-1, 1)         
            Z2 = averagerate[2:].reshape(-1, 1)          

            obs = len(X)

            onevec = np.ones((obs, 1))
            M = np.identity(obs) - (onevec @ onevec.T) / obs

            y = M @ y
            X = M @ X
            Z1 = M @ Z1
            Z2 = M @ Z2

            Z_np = np.column_stack((np.array(Z1), np.array(Z2)))

            df_Z = pd.DataFrame(Z_np)
            df_Z = (gram_schmidt_process(df_Z[[0, 1]])) * np.sqrt(obs)
            Z = np.vstack((df_Z[0].values, df_Z[1].values)).T
            W = np.hstack((y, X))

            J_init = ScoreTest()
            KP_init = ScoreTest()

            lags = 4
            J_test = J_init.score_test(y=y, X=X, Z=Z, W=None, method="2sls", errors="hac", lags=lags, no_constant=True)
            KP_test = KP_init.score_test(y=y, X=X, Z=Z, W=None, method="liml", errors="hac", lags=lags, no_constant=True)

            TSLS = J_test.coefficients[0][0]
            LIML = KP_test.coefficients[0][0]
            J = J_test.statistic
            KP = KP_test.statistic

            # Compute resiudals for effective F statistic
            Pi_TSLS = np.linalg.lstsq(Z, X, rcond = None)[0]
            v_TSLS = X - Z @ Pi_TSLS

            eff, effDOF =  effective_F(X, Z, v_TSLS, lags)
            cv = critical_value(effDOF, 10)

            new_row = {
                'Country': country,
                'n': obs,
                'F': eff[0][0],
                'cv': cv,
                '2SLS': TSLS,
                'LIML': LIML,
                'J': J,
                'KP': KP
            }
            results_list.append(new_row)

        results_table = pd.DataFrame(results_list)

        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
            print(f"Created directory '{OUTPUT_DIR}'")
        table_name = f"table_{table}.csv"
        csv_path = os.path.join(OUTPUT_DIR, table_name)
        results_table.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()

