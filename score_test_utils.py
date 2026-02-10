"""
Utility functions and classes for overidentification testing
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Union

"""
Functions for parameterr estimation
"""

def compute_2sls(
        y: np.ndarray, 
        X: np.ndarray, 
        Z: np.ndarray
    ) -> tuple:

    """Compute the 2SLS estimator"""
    pi_hat = np.linalg.pinv(Z.T @ Z) @ (Z.T @ X)
    X_hat = Z @ pi_hat
    beta_hat = np.linalg.pinv(X_hat.T @ X_hat) @ (X_hat.T @ y)
    u_hat = y - X @ beta_hat
    M_X_hat = np.eye(len(y)) - X_hat @ np.linalg.pinv(X_hat.T @ X_hat) @ X_hat.T

    return beta_hat, u_hat, M_X_hat
        
def compute_liml(
        y: np.ndarray, 
        X: np.ndarray, 
        Z: np.ndarray
    ) -> tuple:

    """Compute the LIML estimator"""
    n = len(y)
    Y = np.column_stack([y, X])
    sigY = Y.T @ Y / n
    sigY12 = sqrtm(sigY)
    YPZY = Y.T @ Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ Y
    eig_mat = np.linalg.solve(sigY12, YPZY @ np.linalg.solve(sigY12.T, np.eye(sigY12.shape[0]))) / n
    min_eig = min(np.linalg.eigvals(eig_mat).real)
    
    X_hat = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T @ X
    beta_hat = np.linalg.pinv(X_hat.T @ X - min_eig * (X.T @ X)) @ (X_hat.T @ y - min_eig * (X.T @ y))
    u_hat = y - X @ beta_hat
    
    M_u_hat = np.eye(n) - u_hat @ np.linalg.pinv(u_hat.T @ u_hat) @ u_hat.T
    pi_liml = np.linalg.pinv(Z.T @ M_u_hat @ Z) @ Z.T @ M_u_hat @ X
    X_hat_liml = Z @ pi_liml
    M_X_hat = np.eye(n) - X_hat_liml @ np.linalg.pinv(X_hat_liml.T @ X_hat_liml) @ X_hat_liml.T
    
    return beta_hat, u_hat, M_X_hat

"""
Classes of variance estimator
"""

class VarianceEstimator:
    """Standard variance covariance matrix estimator"""

    def compute(
            self, 
            residuals: np.ndarray, 
            MXZ2: np.ndarray, 
            **kwargs
        ) -> np.ndarray:
        raise NotImplementedError


class HomoskedasticVariance(VarianceEstimator):
    """Homoskedastic variance estimator"""

    def compute(
            self, 
            residuals: np.ndarray, 
            MXZ2: np.ndarray, 
            **kwargs
        )-> np.ndarray:

        n = len(residuals)
        sigma2_hat = np.sum(residuals ** 2) / n
        return (MXZ2.T @ MXZ2) * sigma2_hat


class HeteroskedasticVariance(VarianceEstimator):
    """White heteroskedasticity-robust variance estimator"""

    def compute(
            self, 
            residuals: np.ndarray, 
            MXZ2: np.ndarray, 
            **kwargs
        ) -> np.ndarray:

        residuals2 = residuals ** 2
        return MXZ2.T @ np.diag(residuals2.flatten()) @ MXZ2


class NeweyWestVarianceEstimator(VarianceEstimator):
    """Newey-West Heteroskedasticity and autocorrelation robust variance estimator"""

    def _get_lag_length(
            self, 
            n: int, 
            method: str
        ) -> int:
        """Lags for Newey-West variance estimator"""

        if method == "rot":
            return int(4 * (n / 100) ** (2 / 9))
        elif method == "plug-in":
            return int((4 / 3) * (n / 100) ** (1 / 4))
        else:
            try:
                return int(method)
            except ValueError:
                raise ValueError("Please enter 'rot', 'plug-in' or a positive integer for the `lags` argument")

    def compute(
            self, 
            residuals: np.ndarray, 
            MXZ2: np.ndarray,
            lags: Union[str, int] = "rot", 
            **kwargs
        ) -> np.ndarray:
        """Compute the variance estimator"""

        n = len(residuals)
        V = MXZ2.T @ np.diag(residuals.flatten() ** 2) @ MXZ2
        L = self._get_lag_length(n, lags)

        if L > 0:
            for l in range(1, L + 1):
                w_l = 1 - l / (L + 1)
                for t in range(l, n):
                    gamma = w_l * residuals[t] * residuals[t - l]
                    V += gamma * (MXZ2[t:t + 1].T @ MXZ2[t - l:t - l + 1] +
                                  MXZ2[t - l:t - l + 1].T @ MXZ2[t:t + 1])
        return V


class ClusterVariance(VarianceEstimator):
    """Cluster robust variance estimator"""

    def compute(
            self, 
            residuals: np.ndarray, 
            MXZ2: np.ndarray,
            cluster_var: np.ndarray, 
            **kwargs
        ) -> np.ndarray:

        if cluster_var is None:
            raise ValueError("Cluster variable must be provided for cluster-robust variance errors")

        n = len(residuals)
        clusters = np.unique(cluster_var)
        n_clusters = len(clusters)
        df = MXZ2.shape[1]
        V = np.zeros((df, df))

        for cluster in clusters:
            cluster_idx = np.where(cluster_var == cluster)[0]
            MXZ2_c = MXZ2[cluster_idx]
            residuals_c = residuals[cluster_idx]
            V += MXZ2_c.T @ residuals_c @ residuals_c.T @ MXZ2_c

        V += V * n_clusters / (n_clusters - 1)

        return V