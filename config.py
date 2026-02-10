## ===================================================
## DIRECTORIES/FILES (DO NOT CHANGE)
## ===================================================

SIMULATIONS_FOLDER = "simulations"
APPLICATIONS_FOLDER = "empirical_applications"
YOGO_FOLDER = "yogo2004"
POZZI_FOLDER = "pozzi2022"
OUTPUT_DIR = 'output'
SIMULATION_CSV_FILE = 'simulation_metrics.csv'
DATA_DIR = 'data'
DATA_FILE_POZZI = 'data.txt'

## ===================================================
## WHICH PARTS OF CODE TO RUN
## ===================================================

RUN_SIMS = True
RUN_YOGO = False
RUN_POZZI = False
# Cleaned code available, set to True to replicate the
# data cleaning process for the Pozzi (2022) dataset
CLEAN_RAW_POZZI_DATA = False

## ===================================================
## SIMULATION PARAMETERS
## ===================================================

# Which simulations to run
RUN_MU2_VARYING = True  # Figures 4.1 and C.1, and Tables D.1 and D.2
RUN_RHO_VARYING = False  # Figures 4.2 and C.2
RUN_POWER = False       # Figure 2.1

# Simulation parameters
NUM_SIMS = 20000     # Simulation repetitions
N = 120              # Number of observations
KX = 1               # Number of endogenous regressors
BETA = 0             # Structural coefficient of interest
KZ_VALUES = [2, 4]   # Number of instruments
EXPERIMENTS = [1, 2] # Heteroskedasticity design

# Parallel processing
N_JOBS = -1   # -1 uses all available cores
VERBOSE = 10  # Track progress