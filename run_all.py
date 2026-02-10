import sys
import os
from pathlib import Path
import traceback
from config import *

def main():
    # Store original directory
    original_dir = os.getcwd()
    project_root = Path(__file__).parent.absolute()
    
    # Add project root to Python path (for config.py and score_test.py imports)
    sys.path.insert(0, str(project_root))
    
    # Add all subdirectories to path so local imports work
    sys.path.insert(0, os.path.join(project_root, SIMULATIONS_FOLDER))
    sys.path.insert(0, os.path.join(project_root, APPLICATIONS_FOLDER, YOGO_FOLDER))
    sys.path.insert(0, os.path.join(project_root, APPLICATIONS_FOLDER, POZZI_FOLDER))
    
    print("=" * 80)
    print("RUNNING ALL ANALYSES")
    print("=" * 80)
    print(f"Project root: {project_root}\n")
    
    try:
        if RUN_SIMS:
            # ============================================================
            # RUNNING SIMULATIONS
            # ============================================================
            print("\n" + "=" * 80)
            print("1. RUNNING SIMULATIONS")
            print("=" * 80)
            
            # Change directory for simulations
            os.chdir(os.path.join(project_root, SIMULATIONS_FOLDER))

            # Import and run
            import simulations 
            import convert_simulation_csv_to_latex
            simulations.main()
            convert_simulation_csv_to_latex.main()
        
        if RUN_YOGO:
            # ============================================================
            # RUNNING YOGO (2004) APPLICATION
            # ============================================================
            print("\n" + "=" * 80)
            print("2. RUNNING YOGO (2004) EMPIRICAL APPLICATION")
            print("=" * 80)
            
            # Change directory for Yogo (2004) dataset
            os.chdir(os.path.join(project_root, APPLICATIONS_FOLDER, YOGO_FOLDER))

            # Import and run
            from empirical_applications.yogo2004 import yogo04, convert_yogo_csv_to_latex
            yogo04.main()
            convert_yogo_csv_to_latex.main()
        
        if RUN_POZZI:
            # ============================================================
            # RUNNING POZZI (2022) APPLICATION
            # ============================================================
            print("\n" + "=" * 80)
            print("3. RUNNING POZZI (2022) EMPIRICAL APPLICATION")
            print("=" * 80)
            
            # Change directory for Pozzi (2022) dataset
            os.chdir(os.path.join(project_root, APPLICATIONS_FOLDER, POZZI_FOLDER))

            # Clean raw data if required
            if CLEAN_RAW_POZZI_DATA:
                from empirical_applications.pozzi2022.data import construct_cleaned_data
                construct_cleaned_data.main()

            # Import and run
            from empirical_applications.pozzi2022 import pozzi2022, convert_pozzi_csv_to_latex
            pozzi2022.main()
            convert_pozzi_csv_to_latex.main()
        
        # ============================================================
        # SUMMARY
        # ============================================================
        print("\n" + "=" * 80)
        print("ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nOutputs saved to:")
        print(f"  - simulations/output")
        print(f"  - empirical_applications/yogo2004/output")
        print(f"  - empirical_applications/pozzi2022/output")
        
    except Exception as e:
        print(f"\n{'=' * 80}")
        print("ERROR OCCURRED!")
        print("=" * 80)
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    main()