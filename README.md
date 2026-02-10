REPLICATION PACKAGE FOR: 

  Overidentification testing with weak instruments and heteroskedasticity

  Stuart Lane (University of Bristol) & Frank Windmeijer (University of Oxford)

  ArXiv preprint arXiv:2509.21096.

================================================================================

Author: Stuart Lane
Institution: University of Bristol
Email: stuart.lane@bristol.ac.uk
Date: February 2026

This package contains code to replicate the Monte Carlo simulations and 
empirical applications in "Overidentification testing with weak instruments
and heteroskedasticity"

## QUICK START

To replicate all results in the paper, simply run:

    python run_all.py

in the project root folder.

This will execute:
  1. All Monte Carlo simulations (Sections 5, C, and D)
  2. Yogo (2004) empirical application (Section 6)
  3. Pozzi (2022) empirical application (Section A)

Expected total runtime: 2-3 hours (depending on hardware)

All outputs (tables and figures) will be saved to the appropriate directories 
(`output/` folder will be created in each directory).

## SYSTEM REQUIREMENTS

Software used:
- Python 3.14
- Required packages listed in requirements.txt

Hardware:
- Simulations run on standard laptop with these specs:
      Platform:         Windows-11-10.0.26100-SP0
      Processor:        Intel64 Family 6 Model 198 Stepping 2, GenuineIntel
      Machine:          AMD64
      CPU cores:        24 physical, 24 logical
      RAM:              31.46 GB total, 5.74 GB available

- Expected runtime: 2-3 hours for full simulations depending on hardware. This
   runtime can be significantly reduced by reducing `NSIM` in `config.py` to
   produce approximate results.
- Memory: 8GB RAM recommended

Installation:

    pip install -r requirements.txt

# REPLICATION INSTRUCTIONS

OPTION 1: Run everything at once (recommended)
-----------------------------------------------

Simply execute the master script:

    python run_all.py

This automatically runs all three components below in sequence.


OPTION 2: Run individual components
------------------------------------

If you want to run simulations or applications separately, run
the master script `run_all.py` but set the following parameters 
in config to `True`/`False`:

   -`RUN_SIMS`

   -`RUN_YOGO`

   -`RUN_POZZI`

Simulations expected runtime: ~2-3 hours
Yogo expected runtime: ~1-2 minutes
Pozzi expected runtime: ~1-2 minutes


### SIMULATIONS:
(Optional) Configure which simulations to run in `simulations.py`:

   `RUN_MU2_VARYING = True`   (Figures 4.1 and C.1, and Tables D.1 and D.2)

   `RUN_RHO_VARYING = True`   (Figures 4.2 and C.2)

   `RUN_POWER = True`         (Figure 2.1)

The `simulations/output/` folder then stores intermediary .csv files:
- mu2_varying_results.csv
- rho_varying_results.csv 
- power_results.csv     

All figures saved as .pdf files, of the example format:
- exp1_kz2_alpha20_mu2_varying.pdf                                            

Latex tables:
- table_D_1.tex
- table_D_2.tex      

and timing_report.txt for computational information

### YOGO (2004) DATASET:
The `empirical_applications/yogo/output/` folder then stores intermediary .csv files:
- table_1.csv
- table_2.csv

and the latex table
- table_6_1.text

### POZZI (2022) DATASET:
(Optional) To construct the cleaned dataset from raw, run:

    python empirical_applications/pozzi2022/data/construct_cleaned_data.py

The cleaned dataset is already provided for convenience, so `CLEAN_RAW_POZZI_DATA = False`
in `config.py` by default.

The `empirical_applications/pozzi/output/` folder then stores intermediary .csv files:
- table_1.csv
- table_2.csv

and the latex table
- table_A_1.text

## FILE STRUCTURE

```
replication_package/
├── run_all.py                              # Master script 
├── config.py                               # Global configuration
├── LICENSE                                 # License information
├── requirements.txt                        # Python package dependencies
├── score_test.py                           # Main robust score test class
├── score_test_utils.py                     # Utilities for robust score testing
├── README.txt                              # This file
│
├── simulations/
│   ├── simulations.py                      # Monte Carlo simulation runner
│   ├── simulation_utils.py                 # DGP generation, plotting functions, etc.
│   ├── convert_simulation_csv_to_latex.py  # Converts .csv to .text
│   └── output/                             # PDF figures and LaTeX tables (folder produced in code)
│
└── empirical_applications/
    ├── yogo2004/
    │   ├── yogo04.py                       # Yogo (2004) replication
    |   ├── application_utils.py            # Data cleaning, effective F-statistics, etc.
    │   ├── convert_yogo_csv_to_latex.py    # Converts .csv to .tex                       
    │   ├── data/  
    |   |   ├── AULQ.txt                    # Data files
    |   |   ...                  
    │   |   └── USAQ.txt                    # Data files
    │   └── output/                         # CSV results and LaTeX tables (folder produced in code)
    │
    └── pozzi2022/
        ├── pozzi2022.py                    # Pozzi (2022) replication
        ├── application_utils.py            # Data cleaning, effective F-statistics, etc.
        ├── convert_pozzi_csv_to_latex.py   # Converts .csv to .tex
        ├── data/                           
        |   ├── data.txt                    # Main data file
        |   ├── JSTdatasetR6.xlsx           # Raw data file
        |   └── construct_cleaned_dataset   # Script to produce cleaned data file
        └── output/                         # CSV results and LaTeX tables (folder produced in code)
```

## CONFIGURATION

The file config.py contains global settings:

- File directories, do NOT change these

# Parameters for running individual elements of the code
- `RUN_SIMS` (default: True)
- `RUN_YOGO` (default: True)
- `RUN_POZZI` (default: True)

# Parameters for running individual elements of simulations
- `RUN_MU2_VARYING` (default: True)  # Figures 4.1 and C.1, and Tables D.1 and D.2
- `RUN_RHO_VARYING` (default: True)  # Figures 4.2 and C.2
- `RUN_POWER` (default: True)        # Figure 2.1

- `NSIM`: Number of Monte Carlo repetitions (default: 20000)
  - Reduce this value for faster approximate results
  - Recommended minimum for testing: 1000
- `N`: Sample size (default: 120)
- `KX`: Number of endogenous regressors (default: 1)
- `BETA`: Value of the coefficient of interest (default: 0)
- `KZ_VALUES`: Grid of number of instruments (default: [2, 4])
- `EXPERIMENTS`: Grid of heteroskedasticity designs ([1, 2])

- `N_JOBS`: How many cores to leave free (default: -1)
- `VERBOSE`: Track progress through simulations (default: every 10 steps)

To modify simulation parameters, edit config.py before running.

## DATA SOURCES

Yogo (2004) data:
- Original source: https://sites.google.com/site/motohiroyogo/research/econometrics?authuser=0

Pozzi (2022) data:
- Original source: https://www.macrohistory.net/database/
- The cleaned dataset is available immediately for convenience. The script to  
  produce the cleaned dataset from the raw dataset downloaded in the link above
  is also available. To run this:

        python empirical_applications/pozzi2022/data/construct_cleaned_dataset.py

  from the project root.

## CONTACT

If you find any bugs with the code/any issues running etc, please contact: 
  
  stuart.lane@bristol.ac.uk

## CITING THIS CODE

If you use this code, please cite:

Lane, S., & Windmeijer, F. (2025). Overidentification testing with weak 
instruments and heteroskedasticity. ArXiv preprint arXiv:2509.21096.

## FURTHER REFERENCES

Pozzi, L. (2022). Housing returns and intertemporal substitution in consumption:
estimates for industrial economies. Working Paper. Available at SSRN 4179082.

Yogo, M. (2004). Estimating the elasticity of intertemporal substitution 
when instruments are weak. Review of Economics and Statistics, 86(3), 797-810.

## VERSION HISTORY

v1.0 (February 2026)