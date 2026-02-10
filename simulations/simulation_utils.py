import numpy as np
from scipy.special import gamma
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from score_test import ScoreTest
from itertools import product
from joblib import Parallel, delayed
from typing import Dict, Any
import os
import platform
import sys
import multiprocessing
from datetime import datetime
from scipy.signal import savgol_filter

from config import *

### ==============================================================================
### DATA GENERATING PROCESS, ESTIMATOR, METRICS ETC. FUNCTIONS
### ==============================================================================

def get_pi_vector(
        experiment: int,
        n: int,
        kz: int,
        alpha: float, 
        mu2
    ) -> np.ndarray:
    """Compute Pi vector"""
    
    if experiment == 1:
        m1 = 2**alpha * gamma(alpha + 0.5) / np.sqrt(np.pi)
        m2 = 2**(alpha + 1) * gamma(alpha + 1.5) / np.sqrt(np.pi)
        omega_trace = m2 + (kz - 1) * m1
        
    elif experiment == 2:
        omega_trace = kz * (1 + alpha**2) * np.exp(alpha**2 * kz / 2)
    
    c = np.sqrt(mu2 * omega_trace / (n * kz**2))
    
    Pi_vector = (c * np.ones(kz)).reshape(-1, 1)
    
    return Pi_vector

def generate_data(
        experiment: int,
        n: int,
        kz: int,
        alpha: float,
        beta: float, 
        Pi_vector: np.ndarray, 
        sigma_uv: np.ndarray,
        delta: float = 0.0
    ) -> tuple:

    """Data generating process for each simulation loop"""
    z = np.random.normal(0, 1, (n, kz))
    eps = np.random.multivariate_normal(mean=[0, 0], cov=sigma_uv, size=n)

    if experiment == 1:
        het_factor = (abs(z[:, 0])) ** alpha
        eps1 = het_factor * eps[:, 0]
        eps2 = het_factor * eps[:, 1]
        
    elif experiment == 2:
        za = z * alpha
        zasum = za.sum(axis=1)
        eps1 = np.exp(zasum / 2) * eps[:, 0]
        eps2 = np.exp(zasum / 2) * eps[:, 1]

    dd = delta / np.sqrt(n)
    local_alternative = dd * z[:, 0]
    
    u = (eps1 + local_alternative).reshape(-1, 1)
    v = eps2.reshape(-1, 1)

    x = z @ Pi_vector + v
    y = x * beta + u

    X = np.hstack([np.ones(n).reshape(-1, 1), x]) 
    Z = np.hstack([np.ones(n).reshape(-1, 1), z]) 

    return y, X, Z


def estimator_metrics(
        df: pd.DataFrame, 
        col: str, 
        beta_0: float
    ) -> tuple:

    """Calculate estimator median bias and 90:10 ratio"""
    median_bias = df[col].median() - beta_0
    ninety_ten = df[col].quantile(0.9) - df[col].quantile(0.1)
    return {
        'median_bias': median_bias,
        'ninety_ten': ninety_ten
    }


def rejection_frequency(
        df: pd.DataFrame, 
        col: str, 
        degrees_of_freedom: int, 
        significance_level: float = 0.05
    ) -> float:

    """Calculate the rejection frequency of a test"""
    return ((df[col] > chi2.ppf(1 - significance_level, df=degrees_of_freedom)).astype(int)).mean()


def run_single_simulation(
        experiment: int,
        n: int,
        kz: int,
        alpha: float,
        beta: float, 
        Pi_vector: np.ndarray, 
        sigma_uv: np.ndarray,
        delta: float = 0.0
    ) -> dict:

    """Run a single simulation repetition"""
    y, X, Z = generate_data(experiment, n, kz, alpha, beta, Pi_vector, sigma_uv, delta)

    J_init = ScoreTest()
    KP_init = ScoreTest()
    
    J_test = J_init.score_test(y = y, X = X, Z = Z, method = "2sls", errors = "het", no_constant = True)
    KP_test = KP_init.score_test(y = y, X = X, Z = Z, method = "liml", errors = "het", no_constant = True)
    # Note: no_constant is set to true above, as the constant is already added in generate_data()

    return {
        'TSLS': J_test.coefficients[1][0],
        'LIML': KP_test.coefficients[1][0],
        'J': J_test.statistic,
        'KP': KP_test.statistic
    }

def run_simulations_for_params(
        experiment: int, 
        kz: int, 
        rho: float, 
        alpha: float, 
        mu2: float, 
        n: int, 
        num_sims: int, 
        kx: int, 
        beta: float,
        delta: float = 0.0
    ) -> dict:

    """Run all simulations for a given parameter combination"""
    degs_of_freedom = kz - kx
    sigma_uv = np.array([[1, rho], [rho, 1]])
    Pi_vector = get_pi_vector(experiment, n, kz, alpha, mu2)
    
    # Parallel simulation runs
    results = Parallel(n_jobs=-1)(
        delayed(run_single_simulation)(experiment, n, kz, alpha, beta, Pi_vector, sigma_uv, delta)
        for _ in range(num_sims)
    )
    
    # Convert to DataFrame
    sim_results_df = pd.DataFrame(results)
    
    # Calculate metrics
    TSLS_metrics = estimator_metrics(sim_results_df, 'TSLS', beta)
    LIML_metrics = estimator_metrics(sim_results_df, 'LIML', beta)
    
    parameter_configuration_metrics = {
        'design': experiment,
        'kz': kz,
        'rho': rho,
        'alpha': alpha,
        'mu2': mu2,
        'delta': delta,
        'omega': delta / np.sqrt(n),
        'TSLS_median_bias': TSLS_metrics['median_bias'],
        'LIML_median_bias': LIML_metrics['median_bias'],
        'TSLS_ninety_ten': TSLS_metrics['ninety_ten'],
        'LIML_ninety_ten': LIML_metrics['ninety_ten'],
        'J_reject_10': rejection_frequency(sim_results_df, 'J', degs_of_freedom, significance_level=0.1),
        'J_reject_5': rejection_frequency(sim_results_df, 'J', degs_of_freedom, significance_level=0.05),
        'J_reject_1': rejection_frequency(sim_results_df, 'J', degs_of_freedom, significance_level=0.01),
        'KP_reject_10': rejection_frequency(sim_results_df, 'KP', degs_of_freedom, significance_level=0.1),
        'KP_reject_5': rejection_frequency(sim_results_df, 'KP', degs_of_freedom, significance_level=0.05),
        'KP_reject_1': rejection_frequency(sim_results_df, 'KP', degs_of_freedom, significance_level=0.01),
    }
    
    return parameter_configuration_metrics


def run_simulations(
        simulation_type: str
    ) -> pd.DataFrame:
    """Run the main simulation"""

    print("Starting simulations...")
    print(f"Simulation type: {simulation_type}")

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    if simulation_type == 'mu2_varying':
        rho_values = [0.2, 0.5, 0.95]
        # Mu2 values: finer nearer 0, coarser as strength increases
        mu2_values = (
            [x * 0.5 for x in range(0, 11)] + 
            [x for x in range(6, 33)]
        )
        # Delta grid not used for mu2_varying
        delta_values = [0.0]
        
    elif simulation_type == 'rho_varying':
        mu2_values = [1, 8, 16]
        rho_values = [-0.99] + [round(x * 0.05, 2) for x in range(-19, 20)] + [0.99]
        # Delta grid not used for rho_varying
        delta_values = [0.0]
        
    elif simulation_type == 'power':
        # Power analysis: vary delta, hold mu2 fixed
        # Typical configuration: EXPERIMENTS = [1], KZ_VALUES = [2]
        rho_values = [0.2, 0.5, 0.95]
        mu2_values = [48]
        # Delta grid: finer around 0, coarser in tails
        delta_values = (
            list(range(-30, -3)) + 
            [x * 0.5 for x in range(-5, 6)] + 
            list(range(3, 31))
        )
        
    else:
        raise ValueError("simulation_type must be 'mu2_varying', 'rho_varying', or 'power'")

    # Prepare all parameter combinations
    param_grid = list(product(EXPERIMENTS, KZ_VALUES, rho_values))
    all_params = []
    for experiment, kz, rho in param_grid:
        alpha_values_full = [0.5, 1, 2] if experiment == 1 else [0.05, 0.1, 0.2]
        
        # For rho_varying, only use middle alpha value
        if simulation_type == 'rho_varying':
            alpha_values = [alpha_values_full[1]]
        # For power, use first two alpha values
        elif simulation_type == 'power':
            alpha_values = alpha_values_full[:2] if experiment == 1 else alpha_values_full[:2]
        else:
            alpha_values = alpha_values_full
        
        for i, alpha in enumerate(alpha_values):
            for mu2 in mu2_values:
                for delta in delta_values:
                    all_params.append((experiment, kz, rho, alpha, mu2, N, NUM_SIMS, KX, BETA, delta))

    # Run all parameter combinations in parallel
    print(f"Running {len(all_params)} parameter combinations in parallel...")
    results = Parallel(n_jobs=N_JOBS, verbose=VERBOSE)(
        delayed(run_simulations_for_params)(*params)
        for params in all_params
    )

    # Convert to DataFrame
    df_metrics = pd.DataFrame(results)
    print("Simulation complete!")
    print(f"\nDataFrame shape: {df_metrics.shape}")
    print(df_metrics.head())
    
    return df_metrics

### ==============================================================================
### PRODUCE FIGURES
### ==============================================================================

def generate_figures(
        df_metrics: pd.DataFrame,
        simulation_type: str
    ) -> None:
    """Generate figures from simulation results"""
    
    print("\nGenerating figures...")

    if simulation_type == 'mu2_varying':
        outer_values_fig = [0.2, 0.5, 0.95]
        x_var = 'mu2'
        outer_var = 'rho'
        x_label = r'$\mu^2$'
        x_lim = [0, 32]
        figure_size = (3.15, 2.35)
        
    elif simulation_type == 'rho_varying':
        alpha_values_exp1 = [0.5, 1, 2]
        alpha_values_exp2 = [0.05, 0.1, 0.2]
        outer_values_fig = alpha_values_exp1 + alpha_values_exp2 
        x_var = 'rho'
        outer_var = 'alpha'
        x_label = r'$\rho$'
        x_lim = [-1, 1]
        figure_size = (5.60, 2.7)
        
    elif simulation_type == 'power':
        alpha_values_exp1 = [0.5, 1] 
        outer_values_fig = alpha_values_exp1
        x_var = 'omega'
        outer_var = 'alpha'
        x_label = r'$\omega$'
        x_lim = [-1, 1] 
        figure_size = (3.15, 2.35) 

    param_grid_fig = list(product(EXPERIMENTS, KZ_VALUES, outer_values_fig))

    # Loop over all parameter combinations
    for experiment, kz, outer_value in param_grid_fig:
        
        # For power analysis: ONLY generate exp1, kz=2 figures
        if simulation_type == 'power':
            if experiment != 1 or kz != 2:
                continue
        
        alpha_values = [0.5, 1, 2] if experiment == 1 else [0.05, 0.1, 0.2]

        if simulation_type == 'rho_varying':
            if outer_value not in alpha_values:
                continue
            legend_values = [1, 8, 16]
        elif simulation_type == 'power':
            if outer_value not in alpha_values:
                continue
            legend_values = [0.2, 0.5, 0.95] 
        else:
            legend_values = alpha_values
        
        # Filter data for this combination
        filtered_data = df_metrics[
            (df_metrics["design"] == experiment) & 
            (df_metrics[outer_var] == outer_value) & 
            (df_metrics["kz"] == kz)
        ]
        
        if filtered_data.empty:
            print(f"No data for experiment={experiment}, kz={kz}, {outer_var}={outer_value}")
            continue
        
        # Sort by x_var for smooth curves
        filtered_data = filtered_data.sort_values(x_var)
        
        # Melt data for plotting (same for all simulation types now)
        filtered_data_melted = filtered_data.melt(
            id_vars=['mu2', 'rho', 'alpha', 'design', 'kz', 'omega'] if simulation_type == 'power' else ['mu2', 'rho', 'alpha', 'design', 'kz'],
            value_vars=['J_reject_5', 'KP_reject_5'],
            var_name='test',
            value_name='rejection_rate'
        )
        
        filtered_data_melted['test'] = filtered_data_melted['test'].map({
            'J_reject_5': 'J',
            'KP_reject_5': 'KP'
        })
        
        # Create combined label for legend with clean formatting
        if simulation_type == 'mu2_varying':
            filtered_data_melted['legend_label'] = filtered_data_melted.apply(
                lambda row: f"{row['test']} (α = {int(row['alpha']) if row['alpha'] == int(row['alpha']) else row['alpha']})", axis=1
            )
        elif simulation_type == 'rho_varying':
            filtered_data_melted['legend_label'] = filtered_data_melted.apply(
                lambda row: f"{row['test']} (μ² = {int(row['mu2'])})", axis=1
            )
        elif simulation_type == 'power':
            filtered_data_melted['legend_label'] = filtered_data_melted.apply(
                lambda row: f"{row['test']} (ρ = {row['rho']})", axis=1
            )
        
        # Style parameters
        plt.rcParams.update({
            'font.size': 6,         
            'axes.labelsize': 6,     
            'axes.titlesize': 6,   
            'xtick.labelsize': 6,     
            'ytick.labelsize': 6,     
            'legend.fontsize': 6,     
            'lines.linewidth': 0.5,   
            'axes.linewidth': 0.5,    
            'grid.linewidth': 0.5,    
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8, 
        })
        
        plt.figure(figsize=figure_size)
        
        # Line style customisation
        palette = {}
        dashes = {}
        hue_order = []
        
        for i, legend_val in enumerate(legend_values):
            if simulation_type == 'mu2_varying':
                legend_val_label = int(legend_val) if legend_val == int(legend_val) else legend_val
                label_prefix = 'α = '
            elif simulation_type == 'rho_varying':
                legend_val_label = int(legend_val)
                label_prefix = 'μ² = '
            elif simulation_type == 'power':
                legend_val_label = legend_val
                label_prefix = 'ρ = '
            
            palette[f'J ({label_prefix}{legend_val_label})'] = 'blue'
            palette[f'KP ({label_prefix}{legend_val_label})'] = 'red'
            
            line_style = [(1, 0), (5, 5), (1, 1)][i]
            dashes[f'J ({label_prefix}{legend_val_label})'] = line_style
            dashes[f'KP ({label_prefix}{legend_val_label})'] = line_style
            
            hue_order.append(f'J ({label_prefix}{legend_val_label})')
            hue_order.append(f'KP ({label_prefix}{legend_val_label})')
        
        hue_order = [label for label in hue_order if label.startswith('J')] + \
                    [label for label in hue_order if label.startswith('KP')]
        
        sns.lineplot(
            data = filtered_data_melted,
            x = x_var, 
            y = 'rejection_rate', 
            hue = 'legend_label',
            style = 'legend_label',
            palette = palette,
            dashes = dashes,
            hue_order = hue_order,
            linewidth = 0.8
        )
        
        plt.axhline(y=0.05, color='gray', linestyle='-', linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel('Rejection Frequency')
        
        # Y-axis limits
        y_lim = (0, 1) if simulation_type == 'power' else (0, 0.5)
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        
        # Legend position: upper center for power, upper right otherwise
        legend_loc = 'upper right' if simulation_type == 'mu2_varying' else 'upper center'
        plt.legend(
            loc=legend_loc, 
            frameon=True, 
            edgecolor='gray', 
            fancybox=False,
            prop={'size': 4.5},     
            handlelength=1.5,          
            handletextpad=0.5,         
            labelspacing=0.3,          
            borderpad=0.3,             
            markerscale=0.7            
        )        
        # Tighter layout
        plt.tight_layout()
        
        fig_outer_value = int(outer_value * 100) if outer_var != 'mu2' else outer_value
        fig_path = os.path.join(
            OUTPUT_DIR,
            f'exp{experiment}_kz{kz}_{outer_var}{fig_outer_value}_{simulation_type}.pdf'
        )
        
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {fig_path}")
        
        plt.close()

    print("\nAll figures generated!")

def generate_figures(
        df_metrics: pd.DataFrame,
        simulation_type: str
    ) -> None:
    """Generate figures from simulation results"""
    
    print("\nGenerating figures...")

    if simulation_type == 'mu2_varying':
        outer_values_fig = [0.2, 0.5, 0.95]
        x_var = 'mu2'
        outer_var = 'rho'
        x_label = r'$\mu^2$'
        x_lim = [0, 32]
        figure_size = (3.15, 2.35)
        
    elif simulation_type == 'rho_varying':
        alpha_values_exp1 = [0.5, 1, 2]
        alpha_values_exp2 = [0.05, 0.1, 0.2]
        outer_values_fig = alpha_values_exp1 + alpha_values_exp2 
        x_var = 'rho'
        outer_var = 'alpha'
        x_label = r'$\rho$'
        x_lim = [-1, 1]
        figure_size = (5.60, 2.7)
        
    elif simulation_type == 'power':
        alpha_values_exp1 = [0.5, 1] 
        outer_values_fig = alpha_values_exp1
        x_var = 'omega'
        outer_var = 'alpha'
        x_label = r'$\omega$'
        x_lim = [-1, 1] 
        figure_size = (3.15, 2.35) 

    param_grid_fig = list(product(EXPERIMENTS, KZ_VALUES, outer_values_fig))

    # Loop over all parameter combinations
    for experiment, kz, outer_value in param_grid_fig:
        
        # For power analysis: ONLY generate exp1, kz=2 figures
        if simulation_type == 'power':
            if experiment != 1 or kz != 2:
                continue
        
        alpha_values = [0.5, 1, 2] if experiment == 1 else [0.05, 0.1, 0.2]

        if simulation_type == 'rho_varying':
            if outer_value not in alpha_values:
                continue
            legend_values = [1, 8, 16]
        elif simulation_type == 'power':
            if outer_value not in alpha_values:
                continue
            legend_values = [0.2, 0.5, 0.95] 
        else:
            legend_values = alpha_values
        
        # Filter data for this combination
        filtered_data = df_metrics[
            (df_metrics["design"] == experiment) & 
            (df_metrics[outer_var] == outer_value) & 
            (df_metrics["kz"] == kz)
        ]
        
        if filtered_data.empty:
            print(f"No data for experiment={experiment}, kz={kz}, {outer_var}={outer_value}")
            continue
        
        # Sort by x_var for smooth curves
        filtered_data = filtered_data.sort_values(x_var)
        
        # Melt data for plotting (same for all simulation types now)
        filtered_data_melted = filtered_data.melt(
            id_vars=['mu2', 'rho', 'alpha', 'design', 'kz', 'omega'] if simulation_type == 'power' else ['mu2', 'rho', 'alpha', 'design', 'kz'],
            value_vars=['J_reject_5', 'KP_reject_5'],
            var_name='test',
            value_name='rejection_rate'
        )
        
        filtered_data_melted['test'] = filtered_data_melted['test'].map({
            'J_reject_5': 'J',
            'KP_reject_5': 'KP'
        })
        
        # Create combined label for legend with clean formatting
        if simulation_type == 'mu2_varying':
            filtered_data_melted['legend_label'] = filtered_data_melted.apply(
                lambda row: f"{row['test']} (α = {int(row['alpha']) if row['alpha'] == int(row['alpha']) else row['alpha']})", axis=1
            )
        elif simulation_type == 'rho_varying':
            filtered_data_melted['legend_label'] = filtered_data_melted.apply(
                lambda row: f"{row['test']} (μ² = {int(row['mu2'])})", axis=1
            )
        elif simulation_type == 'power':
            filtered_data_melted['legend_label'] = filtered_data_melted.apply(
                lambda row: f"{row['test']} (ρ = {row['rho']})", axis=1
            )
        
        # Apply Savitzky-Golay smoothing per series
        filtered_data_melted = filtered_data_melted.sort_values(x_var)
        filtered_data_melted['rejection_rate'] = (
            filtered_data_melted
            .groupby('legend_label')['rejection_rate']
            .transform(
                lambda y: savgol_filter(y, window_length=min(7, len(y) if len(y) % 2 == 1 else len(y) - 1), polyorder=3)
                if len(y) >= 5 else y
            )
        )

        # Style parameters
        plt.rcParams.update({
            'font.size': 6,         
            'axes.labelsize': 6,     
            'axes.titlesize': 6,   
            'xtick.labelsize': 6,     
            'ytick.labelsize': 6,     
            'legend.fontsize': 6,     
            'lines.linewidth': 0.5,   
            'axes.linewidth': 0.5,    
            'grid.linewidth': 0.5,    
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8, 
        })
        
        plt.figure(figsize=figure_size)
        
        # Line style customisation
        palette = {}
        dashes = {}
        hue_order = []
        
        for i, legend_val in enumerate(legend_values):
            if simulation_type == 'mu2_varying':
                legend_val_label = int(legend_val) if legend_val == int(legend_val) else legend_val
                label_prefix = 'α = '
            elif simulation_type == 'rho_varying':
                legend_val_label = int(legend_val)
                label_prefix = 'μ² = '
            elif simulation_type == 'power':
                legend_val_label = legend_val
                label_prefix = 'ρ = '
            
            palette[f'J ({label_prefix}{legend_val_label})'] = 'blue'
            palette[f'KP ({label_prefix}{legend_val_label})'] = 'red'
            
            line_style = [(1, 0), (5, 5), (1, 1)][i]
            dashes[f'J ({label_prefix}{legend_val_label})'] = line_style
            dashes[f'KP ({label_prefix}{legend_val_label})'] = line_style
            
            hue_order.append(f'J ({label_prefix}{legend_val_label})')
            hue_order.append(f'KP ({label_prefix}{legend_val_label})')
        
        hue_order = [label for label in hue_order if label.startswith('J')] + \
                    [label for label in hue_order if label.startswith('KP')]
        
        sns.lineplot(
            data = filtered_data_melted,
            x = x_var, 
            y = 'rejection_rate', 
            hue = 'legend_label',
            style = 'legend_label',
            palette = palette,
            dashes = dashes,
            hue_order = hue_order,
            linewidth = 0.8
        )
        
        plt.axhline(y=0.05, color='gray', linestyle='-', linewidth=0.5)
        plt.xlabel(x_label)
        plt.ylabel('Rejection Frequency')
        
        # Y-axis limits
        y_lim = (0, 1) if simulation_type == 'power' else (0, 0.5)
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        
        # Legend position: upper center for power, upper right otherwise
        legend_loc = 'upper right' if simulation_type == 'mu2_varying' else 'upper center'
        plt.legend(
            loc=legend_loc, 
            frameon=True, 
            edgecolor='gray', 
            fancybox=False,
            prop={'size': 4.5},     
            handlelength=1.5,          
            handletextpad=0.5,         
            labelspacing=0.3,          
            borderpad=0.3,             
            markerscale=0.7            
        )        
        # Tighter layout
        plt.tight_layout()
        
        fig_outer_value = int(outer_value * 100) if outer_var != 'mu2' else outer_value
        fig_path = os.path.join(
            SIMULATIONS_FOLDER,
            OUTPUT_DIR, 
            f'exp{experiment}_kz{kz}_{outer_var}{fig_outer_value}_{simulation_type}.pdf'
        )
        
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {fig_path}")
        
        plt.close()

    print("\nAll figures generated!")

### ==============================================================================
### COMPUATIONAL RESOURCE INFORMATION/TIMING REPORTS
### ==============================================================================

# Try to import psutil for detailed system info (optional)
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def save_results(
        df_metrics: pd.DataFrame,
        simulation_type: str
    ) -> None:
    """Save simulation results to CSV"""
    
    csv_path = os.path.join(OUTPUT_DIR, f"{SIMULATION_CSV_FILE.replace('.csv', '')}_{simulation_type}.csv")
    
    df_metrics.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes ({seconds:.1f} seconds)"
    else:
        hours = seconds / 3600
        minutes = (seconds % 3600) / 60
        hours_text = "hour" if hours == 1 else "hours"
        return f"{hours:.0f} {hours_text} {minutes:.0f} minutes"

def get_system_info():
    """Collect system information for reproducibility"""

    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'cpu_count_physical': multiprocessing.cpu_count(),
    }
    
    if HAS_PSUTIL:
        info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        info['ram_total_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
        info['ram_available_gb'] = round(psutil.virtual_memory().available / (1024**3), 2)
    
    return info

def print_system_info(
        info: Dict[str, Any]
    ) -> None:

    """Print system information"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Timestamp:        {info['timestamp']}")
    print(f"Python version:   {info['python_version']}")
    print(f"Platform:         {info['platform']}")
    print(f"Processor:        {info['processor']}")
    print(f"Machine:          {info['machine']}")
    
    if HAS_PSUTIL:
        print(f"CPU cores:        {info['cpu_count_physical']} physical, {info['cpu_count_logical']} logical")
        print(f"RAM:              {info['ram_total_gb']} GB total, {info['ram_available_gb']} GB available")
    else:
        print(f"CPU cores:        {info['cpu_count_physical']}")
        print("(Install psutil for detailed CPU/RAM info: pip install psutil)")
    
    print(f"\nSimulation config:")
    print(f"  NUM_SIMS:       {NUM_SIMS}")
    print(f"  N:              {N}")
    print(f"  N_JOBS:         {N_JOBS} ({'all cores' if N_JOBS == -1 else f'{N_JOBS} cores'})")
    print(f"  EXPERIMENTS:    {EXPERIMENTS}")
    print(f"  KZ_VALUES:      {KZ_VALUES}")
    print("="*60 + "\n")

def save_timing_report(
        system_info: Dict[str, Any], 
        timing_results: Dict[str, float], 
        total_time: float, 
        output_file: str ='timing_report.txt'
    ) -> None:
    """Save timing and system info to file for paper appendix"""
    
    filepath = os.path.join(OUTPUT_DIR, output_file)
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SIMULATION TIMING AND SYSTEM INFORMATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # System info
        f.write("SYSTEM CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Timestamp:           {system_info['timestamp']}\n")
        f.write(f"Python version:      {system_info['python_version']}\n")
        f.write(f"Platform:            {system_info['platform']}\n")
        f.write(f"Processor:           {system_info['processor']}\n")
        f.write(f"Machine:             {system_info['machine']}\n")
        
        if HAS_PSUTIL:
            f.write(f"CPU cores:           {system_info['cpu_count_physical']} physical, "
                   f"{system_info['cpu_count_logical']} logical\n")
            f.write(f"RAM:                 {system_info['ram_total_gb']} GB total, "
                   f"{system_info['ram_available_gb']} GB available\n")
        else:
            f.write(f"CPU cores:           {system_info['cpu_count_physical']}\n")
        
        f.write("\n")
        f.write("SIMULATION PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"NUM_SIMS:            {NUM_SIMS}\n")
        f.write(f"N:                   {N}\n")
        f.write(f"N_JOBS:              {N_JOBS} ({'all cores' if N_JOBS == -1 else f'{N_JOBS} cores'})\n")
        f.write(f"EXPERIMENTS:         {EXPERIMENTS}\n")
        f.write(f"KZ_VALUES:           {KZ_VALUES}\n")
        
        f.write("\n")
        f.write("TIMING RESULTS\n")
        f.write("-" * 70 + "\n")
        
        # Individual simulation timings
        for sim_type, timing in timing_results.items():
            f.write(f"\n{sim_type.upper()}:\n")
            f.write(f"  Time:              {format_time(timing)}\n")
            f.write(f"  Exact:             {timing:.2f} seconds\n")
        
        # Total time
        f.write(f"\nTOTAL TIME:          {format_time(total_time)}\n")
        f.write(f"Exact:               {total_time:.2f} seconds\n")
        
        # Summary
        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY FOR PAPER\n")
        f.write("="*70 + "\n")
        f.write(f"All simulations completed in {format_time(total_time)} on a system with\n")
        if HAS_PSUTIL:
            f.write(f"{system_info['cpu_count_physical']} physical cores ({system_info['cpu_count_logical']} logical) "
                   f"and {system_info['ram_total_gb']} GB RAM,\n")
        else:
            f.write(f"{system_info['cpu_count_physical']} CPU cores,\n")
        f.write(f"using Python {system_info['python_version']} with parallel processing (N_JOBS={N_JOBS}).\n")
        f.write(f"Each simulation type ran {NUM_SIMS} repetitions with sample size N={N}.\n")
        
    print(f"\nTiming report saved to: {filepath}")