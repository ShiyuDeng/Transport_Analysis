"""
Functions for analyzing transport data from experiments.
Last updated: 13/06/2025
Author: [Shiyu Deng]
Email:[dengs@ill.fr] or [sd864@cantab.ac.uk]
"""

import os
import re
import pandas as pd
from pandas import DataFrame as df

import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import itertools
from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from plot_config import cmap, norm, markers

def extract_gpa(filename):
    match = re.search(r'(\d+\.\d+)GPa', filename)
    return float(match.group(1)) if match else 0

# read the orignal data, add resistivity with sample info
def load_data(data_path, h, w, l, label_map=None):
    from sklearn.preprocessing import MinMaxScaler

    all_data = pd.DataFrame()
    files = [f for f in os.listdir(data_path)
             if os.path.isfile(os.path.join(data_path, f))]


    for file in files:
        #gpa = extract_gpa(file)
        file_path = os.path.join(data_path, file)

        #header = pd.read_csv(file_path, sep='\t', skiprows=0, nrows=1, header=None)
        #data = pd.read_csv(file_path, sep='\t', skiprows=1, header=None)
        #data.columns = header.iloc[0]
        
        # --- Locate header line (starts with "Number") ---
        with open(file_path, 'r') as f:
            lines = f.readlines()

        header_line_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith('Number'):
                header_line_index = i
                break

        if header_line_index is None:
            print(f"Warning: No header found in {file}, skipping.")
            continue

        # --- Read data using detected header ---
        data = pd.read_csv(
            file_path, sep='\t', skiprows=header_line_index, header=0
        ).dropna(how='all')  # Drop completely empty rows

        # --- select data 
        selected_columns = ['Elapsed time1 (Sec)', 'Temp1 (K)',
                            'Resistance1 (Ohm)', 'Current1 (A)']
        data = data[selected_columns]
        
        # --- add resistivity
        data['resistivity'] = data['Resistance1 (Ohm)'] * (h * w) / l

        # --- add normalized resistance1 as a new column
        scaler = MinMaxScaler()
        y_col = 'Resistance1 (Ohm)'
        norm_y_col = 'norm_Resistance'

        if data[y_col].notna().sum() > 1:
            data[norm_y_col] = scaler.fit_transform(data[y_col].values.reshape(-1, 1)).flatten()
        else:
            data[norm_y_col] = None  # or fill with np.nan
        
        data = data.reset_index(drop=True)

        if label_map and file in label_map:
            pressure, run_label = label_map[file]
            data['Pressure(GPa)'] = float(pressure)
            data['RunLabel'] = run_label
            print(f"{file} has been read as {pressure} GPa, {run_label}")
        else:
            print(f"Warning: No label found for {file}, skipping.")
            continue

        data['SourceFile'] = file
        all_data = pd.concat([all_data, data], axis=0, ignore_index=True)
    return all_data

def plot_current(all_data, x, y, savefile=False):
    from matplotlib.ticker import FuncFormatter
    def sci_format(x, _): return "{:.1e}".format(x)
    
    ######
    #pressure_values = sorted(all_data['Pressure(GPa)'].unique())
    #num_rows = (len(pressure_values) + 1) // 2
    #num_cols = min(len(pressure_values), 2)
    #######

    unique_combos = all_data[['Pressure(GPa)', 'RunLabel']].drop_duplicates()
    unique_combos = unique_combos.sort_values(by=['Pressure(GPa)', 'RunLabel']).reset_index(drop=True)
    num_plots = len(unique_combos)
    num_cols = min(num_plots, 3)
    num_rows = (num_plots + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, num_rows * 3))
    axes = axes.flatten()

    pressure_values = all_data['Pressure(GPa)'].unique()

    for i, (pressure, run_label) in enumerate(unique_combos.values):
        subset = all_data[
            (all_data['Pressure(GPa)'] == pressure) &
            (all_data['RunLabel'] == run_label)
        ]

        ax = axes[i]
        color = cmap(norm(pressure))
        marker = markers[i % len(markers)]
        
        # Main plot (resistance or resistivity)
        ax.scatter(subset[x],subset[y],
                   c=[color],
                   marker=marker, label=f'{pressure} GPa - {run_label}',
                   s=2, alpha=0.7
                   )

        # Twin y-axis for current
        ax2 = ax.twinx()
        ax2.scatter(subset[x], subset['Current1 (A)'],
                    c='gray', s=0.5, alpha=0.5)
            
        ax2.set_ylabel('Current (A)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.yaxis.set_major_formatter(FuncFormatter(sci_format))
        ax2.grid(False)
            
        ax.set_xlim(0, 300)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(loc='best')
        ax.grid(True)

    # Hide unused subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    if savefile:
        filename = f"current.png"
        plt.savefig(filename, dpi=600, transparent=True)
        print(f"Saved plot to: {filename}")

    plt.show(block=False)
    plt.pause(0.001)


########## check cooling down/warming up rate for each pressure point ###########
def plot_cooling_rate(all_data, pressure, Tmin, Tmax, Tmin_fit, Tmax_fit, x, y,fit_slope=False, savefile=False):
    from scipy.stats import linregress

    # Filter by pressure
    Pdata = all_data[all_data['Pressure(GPa)'] == pressure]
    if Pdata.empty:
        print(f"No data found for {pressure} GPa")
        return None

    slope_results = {}
    run_labels = Pdata['RunLabel'].unique()

    plt.figure(figsize=(6, 4))
    print(f"number of scans: {len(run_labels)}")
    for i, run_label in enumerate(run_labels):
        print(f"{i}, {run_label}")
        data = Pdata[Pdata['RunLabel'] == run_label]
        data = data[(data[y] >= Tmin) & (data[y] <= Tmax)]
        if data.empty:
            continue
            
        x_data = data[x].values
        y_data = data[y].values
        marker = markers[i % len(markers)]
        color = cmap(norm(pressure))
        print(f"Temperature change during {pressure:.1f}GPa_{run_label}_{Tmin}-{Tmax}K")
        plt.scatter(x_data, y_data, s=5, alpha=0.7,
                    marker=marker, color=color,
                    label=f'{pressure:.1f}GPa, {run_label}')

        if fit_slope:
            # Select data within temperature range
            data = data[(data[y] >= Tmin_fit) & (data[y] <= Tmax_fit)]
            x_data = data[x].values
            y_data = data[y].values
                
            # Perform linear regression: Temp vs Time
            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
            slope_results[run_label] = slope
            print(f"Cooling rate at {pressure} GPa between {Tmin_fit}–{Tmax_fit} K: {slope: .3f} K/s")
            print(f"Cooling rate at {pressure} GPa between {Tmin_fit}–{Tmax_fit} K: {60*slope: .3f} K/min")
            plt.plot(x_data, intercept + slope * x_data,
                     color='black',ls='-.', lw=0.5,
                     label=f'Fit: slope={60*slope: .3f} K/min')

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Cooling Rate @ {pressure} GPa, {run_label}')
        plt.legend(fontsize=10)
        plt.grid(True)
            
        if savefile:
            filename = f"{pressure:.1f}GPa_{run_label}_{Tmin}-{Tmax}K_TempChange.png"
            plt.savefig(filename, dpi=600, transparent=True)
            print(f"Saved plot to: {filename}")  
        plt.show()
        #plt.show(block=False)
        #plt.pause(0.001)
        
    return slope_results
#################################################


######## RRR (Residual Resistivity Ratio) #######
########     to assess sample quality     #######
def compute_RRR(data, x='Temp1 (K)', y='resistivity'):
    """
    Compute Residual Resistivity Ratio (RRR) for a subset of data.

    Returns:
        RRR value (float),
        Tmin, Rmin,
        Tmax, Rmax
    """
    Tmin_idx = data[x].idxmin()
    Tmax_idx = data[x].idxmax()

    Tmin = data.loc[Tmin_idx, x]
    Tmax = data.loc[Tmax_idx, x]
    Rmin = data.loc[Tmin_idx, y]
    Rmax = data.loc[Tmax_idx, y]

    RRR_value = Rmax / Rmin if Rmin != 0 else float('inf')
    print(f"RRR = R({Tmax:.1f} K) / R({Tmin:.1f} K) = {Rmax:.2e} / {Rmin:.2e} = {RRR_value:.2e}")
    return RRR_value, round(Tmin, 2), round(Rmin, 2), round(Tmax, 2), round(Rmax, 2)
################################################



#################################################
def plot_transport_data(all_data, x, y, fwidth=6, fheight=5,
                        savefile=False, transport_title=None):
    from sklearn.preprocessing import MinMaxScaler

    pressures = sorted(all_data['Pressure(GPa)'].unique())
    fig, ax = plt.subplots(figsize=(fwidth, fheight))

    #### plot 
    for i, pressure in enumerate(pressures):
        subdata = all_data[all_data['Pressure(GPa)'] == pressure]
        run_labels = subdata['RunLabel'].unique()

        for j, run_label in enumerate(run_labels):
            marker = markers[j % len(markers)]
            subset = subdata[subdata['RunLabel'] == run_label]
            color = cmap(norm(pressure))
            
            ax.plot(subset[x], subset[y],
                    linestyle='None',
                    #color=color,
                    #markerfacecolor=color,
                    marker=marker, markersize=1.0, markeredgewidth=0.5,
                    fillstyle='top', alpha=0.7,
                    label=f"{pressure} GPa, {run_label}")
            
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(f"{y}")
    ax.set_title(transport_title)

    ax.set_yscale('log')
    ax.set_xlim(0, 300)
    ax.set_ylim(5e-2, 1e8)

    ax.grid(True)

    legend = ax.legend(loc='best',
                       ncol=2, fontsize=12,
                       #title='Pressure (GPa)', title_fontsize=13
                       )
    # Increase marker size in legend only
    for handle in legend.legend_handles: # legendHandles:
        handle.set_markersize(6)   

    plt.tight_layout()
    if savefile:
        filename = f"{transport_title}_{y}.png"
        plt.savefig(filename, dpi=600)
        print(f"Saved figure to: {filename}")
    plt.show()

########################
def plot_Arrhenius(all_data, x, y, target_pressure, T_max, T_min, 
                   temperature_ticks='200,250,300', figwidth=5, figheight=5,
                   savepath='ArrheniusFit.png', savefile=False):       

    # Filter data within the specified temperature range (T_min to T_max)
    target_data = all_data[all_data["Pressure(GPa)"] == target_pressure]
    filtered_data = target_data[(target_data[x] >= T_min) & (target_data[x] <= T_max)]
    x_values = 1 / filtered_data[x]
    y_values = np.log(filtered_data[y])

    ### plot the data
    plt.figure(figsize=(figwidth, figheight))
    plt.scatter(x_values, y_values, s=1, color='blue')
    # plt.scatter(1/target_data[x], np.log(target_data[y]), s=10, color='blue')

    ### set the x,y plotting range
    plt.xlim(1/T_max, 1/T_min)
    plt.ylim([y_values.min()-0.02, y_values.max()])

    x_ticks = [1/T for T in temperature_ticks]

    def fractions(x, pos):
        for T in temperature_ticks:
            if np.isclose(x, 1/T):
               return r'$\frac{1}{' + str(int(T)) + '}$'
        return ''

    ax = plt.gca()
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fractions))

    ######################################################################
    ######Perform a linear fit to extract the slope and intercept #########
    filtered_data = target_data[(target_data[x] >= (T_min+8)) & (target_data[x] <= (T_max-8))]
    x_values = 1 / filtered_data[x]
    y_values = np.log(filtered_data[y])

    slope, intercept = np.polyfit(x_values, y_values, 1)

    # Generate y values for the linear fit line
    y_fit = slope * x_values + intercept

    # Overplot the linear fit line
    plt.plot(x_values, y_fit, color='red') #, label=f'Fit: slope={slope:.4f}')

    kB=1.380649*10**(-23) * 6.241509*10**18 ##Bolzman constant, unit: eV/K
    Eg_value=slope*kB
    print("The estimated band gap E_g is %s eV" %Eg_value)

    # Add the annotation, placing it in a dynamically chosen position
    plt.annotate(r'$E_a \approx $ {:.2f} eV'.format(Eg_value),
                xy=(0.35, 0.8), 
                xycoords='axes fraction', 
                textcoords='offset points', 
                xytext=(0, 0),  # Offset the text a bit for clarity
                ha='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray"))

    ########################################################################
    plt.title(f'Arrhenius Ln Fit for {target_pressure} GPa') 
    plt.xlabel('1/T (1/K)')
    plt.ylabel(r'ln($\rho$)')
    plt.grid(True)
    plt.legend(loc='best', fontsize=10)

    plt.tight_layout()
    if savefile:
        plt.savefig(savepath, dpi=600, transparent=True)
        print(f"Saved figure to: {target_pressure}GPa_ArrheniusFit.{format}")
    
    plt.show()

    return Eg_value

###############################
#### analyze dR/dT to determine the transition temperature ####
def NeelTran(
    all_data, target_pressure, Tm1, Tm2,
    x_col='Temp1 (K)', y_col='Resistance1 (Ohm)',
    window_size=5, subsample_factor=15,
    figwidth=4.5, figheight=6, color='orange',
    savepath='NeelTran.png', savefile=False
):
    """
    Interactive function to plot resistivity and its derivative for a given pressure,
    with smoothing and temperature range selection for the Neel transition.
    """

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))

    # Select data for the chosen pressure
    target_data = all_data[all_data["Pressure(GPa)"] == target_pressure]
    x = target_data[x_col]
    y = pd.Series(target_data[y_col])

    # Plot setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figwidth, figheight), 
                                   sharex=True, gridspec_kw={'hspace': 0})

    # Scatter original data
    ax1.scatter(x, y, marker='x', color = cmap(norm(target_pressure)), alpha=0.5, s=5, label='original data')

    # Smooth data
    y_smooth = y.rolling(window_size, min_periods=1).mean()
    x_subsampled = x.iloc[::subsample_factor]
    y_smooth_subsampled = y_smooth.iloc[::subsample_factor]
    ax1.plot(x_subsampled, y_smooth_subsampled, c='red', alpha=1, lw=0.5, label='smoothed data')

    # Derivative of rho over Temp
    derivative = np.gradient(y_smooth_subsampled, x_subsampled)
    ax2.plot(x_subsampled, derivative, c='red', alpha=0.8, lw=1.0)

    # Restrict to Tm1-Tm2 range for zero crossing search
    mask = (x_subsampled >= Tm1) & (x_subsampled <= Tm2)
    x_inrange = x_subsampled[mask]
    d_inrange = derivative[mask]

    zero_crossings = np.where(np.diff(np.sign(d_inrange)))[0]
    if len(zero_crossings) > 0:
        idx = zero_crossings[0]
        # Use position-based indexing to avoid KeyError
        x0, x1 = np.array(x_inrange)[idx], np.array(x_inrange)[idx+1]
        y0, y1 = np.array(d_inrange)[idx], np.array(d_inrange)[idx+1]
        Tmean = x0 - y0 * (x1 - x0) / (y1 - y0)
        T_err = abs(x1 - x0) / 2
    else:
        print("Err: No zero crossing found in the specified temperature range.")
        Tmean = np.nan
        T_err = np.nan

    # Highlight Neel transition region and annotate
    if not np.isnan(Tmean):
        ax1.axvline(Tmean, color='#1f77b4', linestyle='-.', lw=1.0, label=r'$T_N = {0:.1f} \pm {1:.1f}$ K'.format(Tmean, T_err),)
        ax1.axvspan(Tmean-T_err, Tmean+T_err, color='grey', alpha=0.1, zorder=-1)
        ax2.axvline(Tmean, color='#1f77b4', linestyle='-.', lw=1.0)
        ax2.axvspan(Tmean-T_err, Tmean+T_err, color='grey', alpha=0.1, zorder=-1)
        ax2.annotate(
            r'$T_N = {0:.1f} \pm {1:.1f}$ K'.format(Tmean, T_err),
            xy=(Tmean, 0), xycoords=('data', 'data'),
            xytext=(-120, 30), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='#1f77b4'),
            color='#1f77b4', ha='left'
        )
        ax2.axhline(y=0, linewidth=1.0, ls='-.', color='black')

    # Labels and formatting
    ax1.set_yscale('log')
    ax1.set_ylim(y.min()*0.95, y.max()*1.05)
    ax1.set_ylabel(r'Resistivity $\rho$')
    ax1.legend()
    ax1.grid(True)
    plot_title = os.path.splitext(os.path.basename(savepath))[0]  # Extract title without extension
    ax1.set_title(plot_title)
    #ax2.set_ylim(-derivative.max()*1.02, derivative.max()*1.02)
    ax2.set_ylabel(r'$d\rho/d\rmT$')
    ax2.set_xlabel('T (K)')
    ax2.grid(True)
    plt.xlim(10, 280)

    ax2.set_ylim(auto=True)
    plt.tight_layout()
    if savefile:
        plt.savefig(savepath, dpi=600, transparent=True)
    plt.show()

    return Tmean, T_err