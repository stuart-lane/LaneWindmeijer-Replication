import numpy as np
import pandas as pd

def gram_schmidt_process(
        columns: pd.Series
    ) -> pd.DataFrame:

    """Orthonormalise a matrix via the Gram-Schmidt process"""
    num_columns = columns.shape[1]
    orthonormal_columns = np.zeros((columns.shape[0], num_columns))
    for i in range(num_columns):
        v = columns.iloc[:, i]
        for j in range(i):
            v -= np.dot(v, orthonormal_columns[:, j]) * orthonormal_columns[:, j]
        norm = np.linalg.norm(v)
        if norm > 0:
            orthonormal_columns[:, i] = v / norm       
    return pd.DataFrame(orthonormal_columns, columns = columns.columns)


def newey_west(
        e: np.ndarray, 
        X: np.ndarray, 
        L: int
    ) -> np.ndarray:

    """Calculate the Newey-West heteroskedasticity-autocorrelation-robust variance estimator"""
    N = X.shape[0]
    k = X.shape[1] if len(X.shape) > 1 else 1
    
    # Ensure X is 2D and e is a column vector
    X = X.reshape(N, -1)
    e = e.reshape(-1, 1)  # Make e a column vector
    
    Q = np.zeros((k, k))
    for l in range(L + 1):
        w_l = 1 - l / (L + 1)
        for t in range(l + 1, N):
            if l == 0:
                Q += float(e[t]**2) * np.outer(X[t, :], X[t, :])
            else:
                Q += w_l * float(e[t]) * float(e[t - l]) * (np.outer(X[t, :], X[t - l, :]) + np.outer(X[t - l, :], X[t, :]))
    return Q

def newey_west(
        residuals: np.ndarray, 
        MXZ2: np.ndarray,
        lags: int, 
    ) -> np.ndarray:
    """Compute the variance estimator"""

    n = len(residuals)
    V = MXZ2.T @ np.diag(residuals.flatten() ** 2) @ MXZ2
    L = lags

    if L > 0:
        for l in range(1, L + 1):
            w_l = 1 - l / (L + 1)
            for t in range(l, n):
                gamma = w_l * residuals[t] * residuals[t - l]
                V += gamma * (MXZ2[t:t + 1].T @ MXZ2[t - l:t - l + 1] +
                                MXZ2[t - l:t - l + 1].T @ MXZ2[t:t + 1])
    return V

def effective_F(
        X: np.ndarray, 
        Z: np.ndarray, 
        vhat: np.ndarray, 
        lags: int
    ) -> int:

    """ Calculate the effective F statistic and effective degrees of freedom"""
    Fweight = newey_west(vhat, Z, lags)
    Fweight_tr = np.trace(Fweight)
    eff = (X.T @ Z @ Z.T @ X) / Fweight_tr
    effDOF = ((Fweight_tr ** 2) * 21) / ((np.trace(Fweight.T @ Fweight)) + 20 * Fweight_tr * np.max(np.linalg.eigvals(Fweight)))
    return eff, effDOF


def critical_value(
        Keff: float, 
        tau: float,
        quantile: float = 95, 
        repetitions: float = 10000000
    ) -> float:

    """Simulate the critical value for the effective F statistic"""
    np.random.seed(234435)
    vals = np.random.noncentral_chisquare(Keff, tau * Keff, repetitions) / Keff  
    cv = np.percentile(vals, quantile)
    return cv

def clean_pozzi_data(
        original_df: pd.DataFrame
    ) -> tuple:

    new_column_names = original_df.columns[0].split("\t")
    joined_data = original_df.values

    new_data = []

    for row in range(original_df.shape[0]):
        joined_data = original_df.iloc[row, 0]
        data_list = joined_data.split("\t")
        
        row_data = {column_name: value for column_name, value in zip(new_column_names, data_list)}

        new_data.append(row_data)


    df = pd.DataFrame(new_data)
    df[new_column_names[2:]] = df[new_column_names[2:]].apply(pd.to_numeric, errors = 'coerce')

    df = df.reset_index(drop=True)

    table_header = ['Country', 'F', 'cv', '2SLS', 'LIML', 'J', 'KP']

    df = df[['iso', 'country', 'year', 'r_housing_tr', 'dc']]
    df = df[df['year'] != 1950]

    old_iso = ['DNK', 'DEU', 'JPN', 'NLD', 'PRT', 'ESP', 'CHE', 'GBR']
    new_iso = ['DEN', 'GER', 'JAP', 'NTH', 'POR', 'SPA', 'SWT', 'UK']

    # Belgium, Germany and Japan have missing data problems
    df = df[~df['iso'].isin(['BEL', 'GER', 'JAP'])]

    for iso_idx in range(0, len(old_iso)):
        # df['iso'].replace(old_iso[iso_idx], new_iso[iso_idx], inplace = True)
        df['iso'] = df['iso'].replace(old_iso[iso_idx], new_iso[iso_idx])    

    countries = df['iso'].unique()

    return df, table_header, countries