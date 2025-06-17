"""
Main script for analyzing transport data from experiments.
Functions are stored in separate modules for better organization.
- transport_processing.py
- FermiLiquid_Fit.py
Last updated: 13/06/2025
Author: [Shiyu Deng]
Email:[dengs@ill.fr] or [sd864@cantab.ac.uk]
"""
# main.py
import argparse
import importlib
import matplotlib.pyplot as plt

import os
import sys

from transport_processing import load_data, plot_current, plot_cooling_rate, plot_transport_data, plot_Arrhenius, NeelTran, compute_RRR
from FermiLiquid_Fit import FermiLiquid_T2_fit, plot_fermi_liquid_fit, plot_FermiLiquid_stack, plot_FermiLiquid_offset, plot_coefficient_A

#### Main Code ######
def main(data_path, file_labels, h, w, l, Pressure_check, Tmin, Tmax, Tmin_fit, Tmax_fit, Pressures_RRR,
         Project, fit_slope, metal_pressures, slope_values, offset_values, ytrans,
         current=False, cooling=False, analysis_RRR=False, 
         transport=False, transport_norm=False,
         FermiLiquid=None, Arrhenius=False, MagTran=False,
         savefile=False, printplot=False):

    ###### general style #######
    if printplot:
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
        plt.rc('font', **font)
    
    # read the data
    print(f"Reading data from: {data_path}")
    all_data = load_data(data_path, h, w, l, label_map=file_labels)
    
    # check current during measurement
    if current:
        plot_current(all_data, x='Temp1 (K)', y=ytrans,
                     savefile=savefile)

    # check cooling rate
    if cooling:
        for pressure in Pressure_check:
            rate = plot_cooling_rate(all_data, 
                                     pressure, Tmin, Tmax, Tmin_fit, Tmax_fit, 
                                     x='Elapsed time1 (Sec)', 
                                     y='Temp1 (K)',
                                     fit_slope=fit_slope, savefile=savefile
                                     )

    if analysis_RRR:
        print(f"Residual Resistivity Ration analysis for {Project}:")
        for pressure in Pressures_RRR:
            target_data = all_data[all_data["Pressure(GPa)"] == pressure]

            # Compute RRR
            RRR_value, Tmin, Rmin, Tmax, Rmax = compute_RRR(target_data, x='Temp1 (K)', y='Resistance1 (Ohm)')
            #print(f"Tmin: {Tmin:.2f} K, Rmin: {Rmin:.2f} Ohm, Tmax: {Tmax:.2f} K, Rmax: {Rmax:.2f} Ohm")
            print(f"RRR at {pressure} GPa: {RRR_value:.2e}")
            print("**********")

    # resistance/resistivity plots
    if transport:
        plot_transport_data(all_data,
                            x='Temp1 (K)', y=ytrans,
                            fwidth=6, fheight=5,
                            savefile=savefile,
                            transport_title=f'{Project}_{ytrans}')

    if transport_norm:
        plot_transport_data(all_data,
                            x='Temp1 (K)', y='norm_Resistance',
                            fwidth=6, fheight=5,
                            savefile=savefile,
                            transport_title=f'{Project}_{ytrans}_norm')

    ### analyze Fermi-liquid behaviour
    if FermiLiquid:
        fit_results = {}

        for pressure in metal_pressures:
            print(f"*** Perform Fermi-Liquid fitting for {pressure} GPa low T data.")
            target_data = all_data[all_data["Pressure(GPa)"] == pressure]
            for run in target_data['RunLabel'].unique():
                 subset = target_data[target_data['RunLabel'] == run]
            
                 params, covariance, Trange, error = FermiLiquid_T2_fit(
                     subset, x='Temp1 (K)', y='resistivity',
                     Tmin_range=(5, 30), Tmax_span=(10, 15),
                     debug=False)
                 print(f"*** Final Fitting Results for {pressure} GPa. {run} ***")

                 if params is not None:
                     fit_results[(pressure, run)] = {
                         "params": params,
                         "cov": covariance,
                         "Trange": Trange,
                         "error": error,
                         "data": subset
                     }

                     # Collect coefficient A for plotting
                     
                     if 'plot_individual' in FermiLiquid:
                         plot_fermi_liquid_fit(
                             subset, params=params, Trange=Trange,
                             filetitle=f"{Project}_{pressure}GPa_{run}",
                             x='Temp1 (K)', y='resistivity',
                             savefile=savefile, saveformat='png')
                                   
                 else:
                     print(f"No suitable temperature range found for Pressure {pressure} GPa, run label: {run}.")
        print("Finished analyzing for pressures for metallic phase.")
        
        if 'plot_coefficient' in FermiLiquid:
            print("Plotting Fermi-liquid coefficient A as a function of pressure:")

            plot_coefficient_A(
                fit_results, 
                figwidth=5.5, figheight=4.5,
                savefile=savefile,
                saveformat='png',
                title=f"{Project}_fitted_coeff"
            )

        if 'plot_stacked' in FermiLiquid:
            plot_FermiLiquid_stack(all_data, fit_results,
                            target_pressures=metal_pressures,
                            x='Temp1 (K)', y='resistivity',
                            figwidth=5, figheight_per_plot=1.25,
                            savepath=f'{Project}_FermiLiquid_T2_stack.png')
                       
        if 'plot_offset' in FermiLiquid:
            plot_FermiLiquid_offset(
                fit_results,
                target_pressures=metal_pressures,
                slope_values=slope_values,
                offset_values=offset_values,
                figwidth=8, figheight=6,
                savepath=f'{Project}_FermiLiquid_T2_offset.png')
    
    if Arrhenius:
        print("Performing Arrhenius fitting for insulator phase.")
        # Interactive user input 
        target_pressure = float(input("Enter the target pressure (e.g., 6.0): "))
        T_max = float(input("Enter the Temperature range, T_max: "))
        T_min = float(input("Enter the Temperature range, T_min: "))
        temperature_ticks = input("Enter the temperature x-ticks, e.g. 300,275,250: ")
        temperature_ticks = [float(T) for T in temperature_ticks.split(',')]

        Eg_value=plot_Arrhenius(all_data, x='Temp1 (K)', y=ytrans,
                                target_pressure=target_pressure,
                                T_max=T_max, T_min=T_min,
                                temperature_ticks=temperature_ticks,
                                figwidth=5.5, figheight=4.5,
                                savepath=f'{Project}_{target_pressure}GPa_ArrheniusFit.png',
                                savefile=savefile)
        # Print the estimated band gap value
        if Eg_value is not None:
            print(f"Estimated band gap (Eg) value at {target_pressure} GPa for {Project}: {Eg_value:.2f} eV")
        else:
            print("Error in Arrhenius fitting.")
    
    if MagTran:
        print("Performing Neel transition analysis.")
        # Interactive user input
        target_pressure = float(input("Enter the target pressure (e.g., 6.0): "))
        Tm1 = float(input("Enter the lower bound of Neel transition temperature (e.g., 100): "))
        Tm2 = float(input("Enter the upper bound of Neel transition temperature (e.g., 150): "))
        window_size = int(input("Enter the window size for smoothing (e.g., 5): "))
        subsample_factor = int(input("Enter the subsample factor (e.g., 15): "))

        NeelTran(all_data, target_pressure=target_pressure,
                Tm1=Tm1, Tm2=Tm2,
                x_col='Temp1 (K)', y_col=ytrans,
                window_size=window_size, subsample_factor=subsample_factor,
                figwidth=6, figheight=4.5, color='orange',
                savepath=f'{Project}_{target_pressure}GPa_{Tm1}_{Tm2}K_NeelTran.png', 
                savefile=savefile
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transport Data Analysis")
    parser.add_argument("--input", required=True, help="Input config for specified sample")
    parser.add_argument("-current", action="store_true", help="Plot current vs temperature")
    parser.add_argument("-cooling", action="store_true", help="Analyze cooling rate")
    parser.add_argument("-analysis_RRR", action="store_true", help="Analyze Residual Resistivity Ratio (RRR)")
    parser.add_argument("-transport", action="store_true", help="Plot resistance/resistivity")
    parser.add_argument("-transport_norm", action="store_true", help="Plot normalized resistance")
    parser.add_argument("--FermiLiquid",
                        nargs='+',
                        choices=['plot_coefficient','plot_individual', 'plot_stacked', 'plot_offset'],
                        help="Run Fermi-liquid analysis for low Temp metallic phase. Choose from: plot_coefficient, plot_individual, plot_stacked, plot_offset")
    parser.add_argument("-Arrhenius", action="store_true", help="Arrhenius fitting for insulator")
    parser.add_argument("-MagTran", action="store_true", help="Analyze Neel transition")
    parser.add_argument("-saveplot",  action="store_true", help="save the plot")
    parser.add_argument("-printplot", action="store_true", help="Enable publication plot")

    args = parser.parse_args()

    # Clean and normalize input path
    input_path = os.path.abspath(args.input)
    input_dir, input_file = os.path.split(input_path)
    module_name = os.path.splitext(input_file)[0]

    # Add the directory to sys.path so Python can find the module
    sys.path.insert(0, input_dir)

    # Import the module
    config = importlib.import_module(module_name)

    main(
        data_path=config.data_path,
        file_labels=config.file_labels,
        h=config.h,
        w=config.w,
        l=config.l,
        Pressure_check=config.Pressure_check,
        Tmin=config.Tmin,
        Tmax=config.Tmax,
        Tmin_fit=config.Tmin_fit,
        Tmax_fit=config.Tmax_fit,
        Pressures_RRR=config.Pressures_RRR,
        Project=config.Project,
        fit_slope=config.fit_slope,
        metal_pressures=config.metal_pressures,
        slope_values=config.slope_values,
        offset_values=config.offset_values,
        current=args.current,
        cooling=args.cooling,
        analysis_RRR=args.analysis_RRR,
        transport=args.transport,
        transport_norm=args.transport_norm,
        ytrans=config.ytrans,
        FermiLiquid=args.FermiLiquid,
        Arrhenius=args.Arrhenius,
        MagTran=args.MagTran,
        printplot=args.printplot,
        savefile=args.saveplot
    )
