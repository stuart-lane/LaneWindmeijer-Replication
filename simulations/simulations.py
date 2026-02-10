import time
from simulation_utils import *

def main():    
    # Get system info
    system_info = get_system_info()
    print_system_info(system_info)
    
    # Build list of simulations to run based on config
    simulation_types = []
    if RUN_MU2_VARYING:
        simulation_types.append('mu2_varying')
    if RUN_RHO_VARYING:
        simulation_types.append('rho_varying')
    if RUN_POWER:
        simulation_types.append('power')
    
    if not simulation_types:
        print("No simulation types selected! Set at least one RUN_* flag to True.")
        return
    
    print(f"Running {len(simulation_types)} simulation type(s): {', '.join(simulation_types)}\n")
    
    # Start overall timer
    total_start_time = time.time()
    timing_results = {}
    
    for sim_type in simulation_types:
        print(f"\n{'='*60}")
        print(f"Running {sim_type} simulations")
        print(f"{'='*60}\n")
        
        # Start timer for this simulation type
        sim_start_time = time.time()
        
        # Run simulations
        df_metrics = run_simulations(sim_type)
        
        # Generate figures
        generate_figures(df_metrics, sim_type)
        
        # Save results
        save_results(df_metrics, sim_type)
        
        # Calculate elapsed time for this simulation type
        sim_elapsed = time.time() - sim_start_time
        timing_results[sim_type] = sim_elapsed
        
        print(f"\n{sim_type} simulations complete!")
        print(f"Time elapsed: {format_time(sim_elapsed)}")
    
    # Calculate total elapsed time
    total_elapsed = time.time() - total_start_time
    
    # Print summary
    print("\n" + "="*60)
    print("ALL SIMULATIONS COMPLETE!")
    print("="*60)
    print("\nTIMING SUMMARY:")
    print("-" * 60)
    for sim_type, timing in timing_results.items():
        print(f"  {sim_type:15s}: {format_time(timing)}")
    print("-" * 60)
    print(f"  {'TOTAL':15s}: {format_time(total_elapsed)}")
    print("="*60)
    
    # Save timing report
    save_timing_report(system_info, timing_results, total_elapsed)
    
    print("\nSimulation complete! Check timing_report.txt for reproducibility info.")

if __name__ == "__main__":
    main()