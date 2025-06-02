# main.py
import matplotlib.pyplot as plt
import argparse
import importlib
import matplotlib.colors as mcolors

import os
import sys

#### read input info from info.py
# from input_info import h, w, l, data_path, file_labels, Pressure_check, Tmin, Tmax, Tmin_fit, Tmax_fit, Project, savefile, fit_slope, metal_pressures, slope_values, offset_values

from transport_processing import load_data, plot_current, plot_cooling_rate, plot_transport_data
from FermiLiquid_Fit import FermiLiquid_T2_fit, plot_fermi_liquid_fit, plot_LowT_Metal, plot_FermiLiquid_offset

#### Main Code ######
def main(data_path, file_labels, h, w, l, Pressure_check, Tmin, Tmax, Tmin_fit, Tmax_fit, Project, fit_slope, metal_pressures, slope_values, offset_values, current=False, cooling=False, transport=False, transport_norm=False, FermiLiquid=None, savefile=False, printplot=False):
    ###### general style #######
    if printplot:
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
        plt.rc('font', **font)
    
    # read the data
    print(f"Reading data from: {data_path}")
    all_data = load_data(data_path, h, w, l, label_map=file_labels)
    
    # check current during measurement
    if current:
        plot_current(all_data, x='Temp1 (K)', y='Resistance1 (Ohm)',
                     savefile=savefile)

    # check cooling rate
    if cooling:
        for P in Pressure_check:
            rate = plot_cooling_rate(all_data, 
                                     P, Tmin, Tmax, Tmin_fit, Tmax_fit, markers,
                                     x='Elapsed time1 (Sec)', 
                                     y='Temp1 (K)',
                                     fit_slope=fit_slope, savefile=savefile
                                     )


    # resistance/resistivity plots
    if transport:
        plot_transport_data(all_data,
                            x='Temp1 (K)', y='Resistance1 (Ohm)',
                            fwidth=6, fheight=5,
                            savefile=savefile,
                            transport_title=f'{Project}')

    if transport_norm:
        plot_transport_data(all_data,
                            x='Temp1 (K)', y='norm_Resistance',
                            fwidth=6, fheight=5,
                            savefile=savefile,
                            transport_title=f'{Project}_norm')

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
                     debug=True)
                 print(f"*** Final Fitting Results for {pressure} GPa. {run} ***")

                 if params is not None:
                     fit_results[(pressure, run)] = {
                         "params": params,
                         "cov": covariance,
                         "Trange": Trange,
                         "error": error,
                         "data": subset
                     }
                     
                     if 'plot_individual' in FermiLiquid:
                         plot_fermi_liquid_fit(
                             subset, params=params, Trange=Trange,
                             filetitle=f"{Project}_{pressure}GPa_{run}",
                             x='Temp1 (K)', y='resistivity',
                             savefile=savefile, saveformat='png')
                        
                 else:
                     print(f"No suitable temperature range found for Pressure {pressure} GPa, run label: {run}.")
                     
        print("Finished analyzing for pressures for metallic phase.")

        if 'plot_stacked' in FermiLiquid:
            plot_LowT_Metal(all_data, fit_results,
                            target_pressures=metal_pressures,
                            x='Temp1 (K)', y='resistivity',
                            figwidth=4, figheight_per_plot=1.2,
                            savepath=f'{Project}_FermiLiquid_T2_stack.pdf')
            
        if 'plot_offset' in FermiLiquid:
            plot_FermiLiquid_offset(
                fit_results,
                target_pressures=metal_pressures,
                slope_values=slope_values,
                offset_values=offset_values,
                figwidth=8, figheight=6,
                savepath='{Project}_FermiLiquid_T2_offset.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transport Data Analysis for FePSe3")
    parser.add_argument("--input", required=True, help="Input config for specified sample")
    parser.add_argument("-current", action="store_true", help="Plot current vs temperature")
    parser.add_argument("-cooling", action="store_true", help="Analyze cooling rate")
    parser.add_argument("-transport", action="store_true", help="Plot resistance/resistivity")
    parser.add_argument("-transport_norm", action="store_true", help="Plot normalized resistance")
    parser.add_argument("--FermiLiquid",
                        nargs='+',
                        choices=['plot_individual', 'plot_stacked', 'plot_offset'],
                        help="Run Fermi-liquid analysis for low Temp metallic phase. Choose from: plot_individual, plot_stacked, plot_offset")
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
        Project=config.Project,
        fit_slope=config.fit_slope,
        metal_pressures=config.metal_pressures,
        slope_values=config.slope_values,
        offset_values=config.offset_values,
        current=args.current,
        cooling=args.cooling,
        transport=args.transport,
        transport_norm=args.transport_norm,
        FermiLiquid=args.FermiLiquid,
        printplot=args.printplot,
        savefile=args.saveplot
    )
