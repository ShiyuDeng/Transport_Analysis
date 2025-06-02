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
def plot_cooling_rate(all_data, P, Tmin, Tmax, Tmin_fit, Tmax_fit, x, y,fit_slope=False, savefile=False):
    from scipy.stats import linregress

    # Filter by pressure
    Pdata = all_data[all_data['Pressure(GPa)'] == P]
    if Pdata.empty:
        print(f"No data found for {P} GPa")
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
        print(f"Temperature change during {P:.1f}GPa_{run_label}_{Tmin}-{Tmax}K")
        plt.scatter(x_data, y_data, s=5, alpha=0.7,
                    marker=marker, color=color,
                    label=f'{P:.1f}GPa, {run_label}')

        if fit_slope:
            # Select data within temperature range
            data = data[(data[y] >= Tmin_fit) & (data[y] <= Tmax_fit)]
            x_data = data[x].values
            y_data = data[y].values
                
            # Perform linear regression: Temp vs Time
            slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)
            slope_results[run_label] = slope
            print(f"Cooling rate at {P} GPa between {Tmin_fit}–{Tmax_fit} K: {slope: .3f} K/s")
            print(f"Cooling rate at {P} GPa between {Tmin_fit}–{Tmax_fit} K: {60*slope: .3f} K/min")
            plt.plot(x_data, intercept + slope * x_data,
                     color='black',ls='-.', lw=0.5,
                     label=f'Fit: slope={60*slope: .3f} K/min')

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Cooling Rate @ {P} GPa, {run_label}')
        plt.legend(fontsize=10)
        plt.grid(True)
            
        if savefile:
            filename = f"{P:.1f}GPa_{run_label}_{Tmin}-{Tmax}K_TempChange.png"
            plt.savefig(filename, dpi=600, transparent=True)
            print(f"Saved plot to: {filename}")  
        plt.show()
        #plt.show(block=False)
        #plt.pause(0.001)
        
    return slope_results
#################################################


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
                    color=color,
                    #markerfacecolor=color,
                    marker=marker, markersize=1.0, markeredgewidth=0.5,
                    fillstyle='top', alpha=0.7,
                    label=f"{pressure} GPa")
            
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
    for handle in legend.legendHandles:
        handle.set_markersize(5)   

    plt.tight_layout()
    if savefile:
        filename = f"{transport_title}_{y}.png"
        plt.savefig(filename, dpi=600)
        print(f"Saved figure to: {filename}")
    plt.show()

########################

