"""
Package for overidentification testing in linear IV models
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union
from dataclasses import dataclass
from score_test_utils import (
    HomoskedasticVariance,
    HeteroskedasticVariance,
    NeweyWestVarianceEstimator,
    ClusterVariance,
    compute_2sls,
    compute_liml
)

@dataclass
class TestResults:
    """Container for test results"""
    statistic: float
    p_value: float
    df: int
    coefficients: np.ndarray
    method: str

    @property
    def coef_str(self) -> str:
        """Return formatted coefficient string"""
        if self.coefficients.shape[0] == 1:
            return f"{self.coefficients[0][0]:.3f}"
        else:
            coefs = [f"{val[0]:.3f}" for i, val in enumerate(self.coefficients)]
            return f"({', '.join(coefs)})"

    def __str__(self) -> str:
        return (
            f"Overidentification test results\n"
            f"-------------------------------\n"
            f"Method: {self.method}\n"
            f"Coefficients: {self.coef_str}\n"
            f"Statistic: {self.statistic:.3f}\n"
            f"p-value: {self.p_value:.3f}\n"
            f"Degrees of freedom: {self.df}\n"
        )
    
class VarianceEstimator:
    """Standard variance covariance matrix estimator"""

    def compute(self, residuals: np.ndarray, MXZ2: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


class ScoreTest:
    """Class for conducting overidentifcation tests in linear IV models"""

    def __init__(self):
        self.variance_estimators = {
            "hom": HomoskedasticVariance(),
            "het": HeteroskedasticVariance(),
            "hac": NeweyWestVarianceEstimator(),
            "cluster": ClusterVariance()
        }


    def _prepare_data(self,
            y: Optional[np.ndarray] = None,
            X: Optional[np.ndarray] = None,
            Z: Optional[np.ndarray] = None,
            W: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None,
            no_constant: bool = False
        ) -> tuple:
        """
        Prepare data for overidentification testing.
        """

        # Basic error check
        if y is None or X is None or Z is None:
            raise ValueError("y, X and Z must be provided.")
        
        # Ensure correct shapes
        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else np.asarray(X)
        Z = np.asarray(Z)

        # Handle constant
        if not no_constant:
            if W is None:
                W = np.ones((len(y), 1))
            else:
                W = np.asarray(W)
                has_constant = np.any(np.all(np.abs(W - np.ones((W.shape[0], 1))) < 1e-8, axis=0))
                if not has_constant:
                    W = np.column_stack([np.ones(len(y)), W])

        # Remove duplicate instruments
        _, unique_cols = np.unique(Z, axis=1, return_index=True)
        Z = Z[:, np.sort(unique_cols)]

        # Apply weights if provided
        if weights is not None:
            w_sqrt = np.sqrt(weights)
            y = y * w_sqrt.reshape(-1, 1)
            X = X * w_sqrt.reshape(-1, 1)
            Z = Z * w_sqrt.reshape(-1, 1)
            if W is not None:
                W = W * w_sqrt.reshape(-1, 1)

        return y, X, Z, W


    def _partial_out(
            self, 
            y: np.ndarray, 
            X: np.ndarray, 
            Z: np.ndarray,
            W: Optional[np.ndarray] = None
        )-> tuple:

        """Partial out any exogenous regressors, including constant"""
        if W is not None:
            Q_W = np.linalg.qr(W)[0]
            M_W = np.eye(len(y)) - Q_W @ Q_W.T
            y = M_W @ y
            X = M_W @ X 
            Z = M_W @ Z
        return y, X, Z
        

    def score_test(self,
            y: np.ndarray,
            X: np.ndarray,
            Z: np.ndarray,
            W: np.ndarray = None,
            weights: np.ndarray = None,
            method: str = "2sls",
            errors: str = "hom",
            basmann: bool = False,
            lags: Union[str, int] = "rot",
            cluster_var: np.ndarray = None,
            small_sample: bool = False,
            no_constant: bool = False
        ) -> TestResults:

        """Conduct overidentification test in linear IV models
        
        ---- Paramaters ----

        // Matrix/vector parameters
        y : array_like
            Dependent variable
        X : array_like
            Endogenous regressors
        Z : array_like
            Excluded instruments
        W : arrray_like, optional,
            Included exogenous regressors
        
        // Other parameters
        weights : array_like, optional
            Weights for computing regression coefficients
        method : str, optional
            Method used to calculate regression coefficient ('2sls' or 'liml')
            Default is '2sls'
        errors : str, optional 
            Structure of the variance matrix ('hom', 'het', 'hac' or 'cluster')
            Default is 'hom'
        basmann : bool, optional
            Use Basmann version of the test statistic
            Default is False
        lags : str or int, optional
            Number of lags for HAC variance estimator ('rot', 'plug-in' or user-specified non-negative integer)
            Default is 'rot'
        cluser_var : array_like, optional
            Clustering variable for cluster variance estimator
        small_sample : bool, optional
            Apply small-sample variance estimator correction
            Default is False
        no_constant : bool, optional
            Do not add constant to regression
            Default is False            

        ---- Returns ---- 

        TestResults
            Test results container
        """

        y, X, Z, W = self._prepare_data(
            y = y,
            X = X,
            Z = Z,
            W = W,
            weights = weights,
            no_constant = no_constant
        )

        # Ensure overidentification for computability of test statistic
        n, kx = X.shape[0], X.shape[1]
        kz = Z.shape[1]
        if kz - kx < 1:
            raise ValueError("Model must be overidentified.")
        df = kz - kx

        # Partial out any included exogenous regressors
        y, X, Z = self._partial_out(y, X, Z, W)

        if method.lower() == "2sls":
            beta_hat, u_hat, M_X_hat = compute_2sls(y, X, Z)
        elif method.lower() == "liml":
            beta_hat, u_hat, M_X_hat = compute_liml(y, X, Z)
        else:
            raise ValueError("Method specified must be either '2sls' or 'liml'.")
        
        Z2 = Z[:, -df:]
        MXZ2 = M_X_hat @ Z2

        if not basmann:
            residuals = u_hat
        else:
            MZ = np.eye(n) - Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
            residuals = MZ @ y - MZ @ (X @ beta_hat)

        variance_matrix = self.variance_estimators.get(errors.lower())
        if variance_matrix is None:
            raise ValueError(f"Unknown error structure: {errors}. Please select from 'hom', 'het', 'hac' or 'cluster'.")
        
        variance = variance_matrix.compute(
            residuals = residuals,
            MXZ2 = MXZ2,
            lags = lags,
            cluster_var = cluster_var
        )

        # Compute the score statistic
        score_stat = float(residuals.T @ MXZ2 @ np.linalg.pinv(variance) @ MXZ2.T @ residuals)

        if small_sample:
            score_stat = n * score_stat / (n - kz)


        p_value = 1 - stats.chi2.cdf(score_stat, df)

        return TestResults(
            statistic=score_stat,
            p_value=p_value,
            df=df,
            coefficients=beta_hat,
            method=f"{'2SLS' if method.lower() == '2sls' else 'LIML'}"
        )
