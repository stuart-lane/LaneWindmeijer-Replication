import os
import numpy as np
import pandas as pd
from scipy import stats
from score_test import ScoreTest
from application_utils import *
from config import *

text_files = os.listdir(DATA_DIR)

table_header = ['Country', 'n', 'F', 'cv', '2SLS', 'LIML', 'J', 'KP']
tables = [1, 2]

def main(): 

    for table in tables:

        results_list = []

        y_col = 'DC' if table == 1 else 'RRF'
        x_col = 'RRF' if table == 1 else 'DC' 

        for text_file in text_files:

            original_df = pd.read_csv(os.path.join(DATA_DIR, text_file))

            new_column_names = original_df.columns[0].split("\t")
            joined_data = original_df.values

            new_data = []

            for row in range(original_df.shape[0]):
                joined_data = original_df.iloc[row, 0]
                data_list = joined_data.split("\t")
                
                row_data = {column_name: value for column_name, value in zip(new_column_names, data_list)}

                new_data.append(row_data)


            df = pd.DataFrame(new_data)
            df[new_column_names[0]] = df[new_column_names[0]].apply(pd.to_datetime, errors = 'coerce')
            df[new_column_names[1:]] = df[new_column_names[1:]].apply(pd.to_numeric, errors = 'coerce')

            df = df.iloc[2:]
            df = df.reset_index(drop = True)

            if text_file != 'USAQ.txt':
                lags = 4
            else:
                lags = 6

            obs = len(df)
            onevec = np.ones((obs, 1))
            M = np.identity(obs) - (onevec @ onevec.T) /obs

            model_vars = ['dc', 'rr', 'rrf', 'z1', 'z2', 'z3', 'z4']
            new_model_vars = [var.upper() for var in model_vars]

            df[new_model_vars] = df[model_vars].apply(lambda x: M @ x)

            df[new_model_vars[-4:]] = gram_schmidt_process(df[new_model_vars[-4:]])
            df[new_model_vars[-4:]] = df[new_model_vars[-4:]] * np.sqrt(obs)
                
            country_name = text_file[:-5]
            if not len(country_name) == 3:
                country_name = ''.join([country_name, ' '])

            y = df[y_col].values
            X = df[x_col].values
            Z = df[new_model_vars[-4:]].values

            J_init = ScoreTest()
            KP_init = ScoreTest()

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
                'Country': country_name,
                'n': obs,
                'F': eff,
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