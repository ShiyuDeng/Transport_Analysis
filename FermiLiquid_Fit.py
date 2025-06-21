"""
Functions for analyzing transport data in metallic phases.
Last updated: 13/06/2025
Author: [Shiyu Deng]
Email:[dengs@ill.fr] or [sd864@cantab.ac.uk]
"""

import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plot_config import cmap, norm, markers
########
def Landau_T2(T_squared, rho0, A):
    return rho0 + A * T_squared

def Landau_resistivity(T, rho0, A):
    return rho0 + A * T**2

########
def FermiLiquid_T2_fit(data, x='Temp1 (K)', y='resistivity',
                       Tmin_range=(5, 30), Tmax_span=(10, 15), debug=False):
    
    rho0_estimate =data[y].min()
    best_params, best_covariance = None, None
    best_error = float('inf')
    best_T_range = None

    for Tmin in range(*Tmin_range):
        for Tmax in range(Tmin+Tmax_span[0], Tmin+Tmax_span[1]+1):
            mask = (data[x] > Tmin) & (data[x] < Tmax)
            filtered_data=data[mask]

            if len(filtered_data) < 2:
                continue
            
            # Use curve_fit to fit the data
            initial_guess = [rho0_estimate, 1.0]
            params, covariance = curve_fit(Landau_resistivity,
                                           filtered_data[x],
                                           filtered_data[y],
                                           p0=initial_guess)

            # Calculate the error of the fit
            residuals = filtered_data[y] - Landau_resistivity(filtered_data[x], *params)
            error = np.mean(residuals**2)
            if debug:
                print(f"  Temp Range: {Tmin}-{Tmax} K")
                print(f"  Err: {error}")
                print("***********************")

            # Update best fit parameters if the error is lower
            if error < best_error:
                best_error = error
                best_params = params
                best_covariance = covariance
                best_T_range = (Tmin, Tmax)

    #### print out the final fiting results
    print(f"  rho0 = {best_params[0]:.4e} ± {np.sqrt(best_covariance[0, 0]):.2e}")
    print(f"  A    = {best_params[1]:.4e} ± {np.sqrt(best_covariance[1, 1]):.2e}")
    print(f"  Temp Range: {best_T_range[0]}–{best_T_range[1]} K")
    print(f"  Error: {best_error}")
                
    return best_params, best_covariance, best_T_range, best_error


def plot_fermi_liquid_fit(data, params, Trange, filetitle, x='Temp1 (K)', y='resistivity', savefile=True, saveformat='png'):
    
    T_fit = np.linspace(Trange[0], Trange[1], 50**2)
    fit_curve = Landau_resistivity(T_fit, *params)

    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(data[x]**2, data[y],
                s=5, alpha=0.3, color='blue')
    plt.plot(T_fit**2, fit_curve, color='black', lw=1.5)

    ylim_max = max(data[(data[x] >= Trange[0]) & (data[x] <= Trange[1])][y])
    ylim_min = min(fit_curve)
    
    plt.ylim(ylim_min, ylim_max)
    plt.xlim(0, 45**2)

    plt.axvline(x=Trange[0]**2, color='cyan', linestyle='-.', lw=0.5,
                label=r'$T_{min}$'+f': {Trange[0]} K')
    plt.axvline(x=Trange[1]**2, color='red', linestyle='-.', lw=0.5,
                label=r'$T_{max}$'+f': {Trange[1]} K')

    plt.xlabel(r"$T^2$ (K$^2$)")
    plt.ylabel(r"Resistivity $\rho$ ($\Omega \cdot$m)")
    plt.title(filetitle)
    plt.legend()
    plt.tight_layout()
    
    if savefile:
        plt.savefig(f'{filetitle}_T2fit.{saveformat}', dpi=600, transparent=True)
    plt.show()


###################
def plot_coefficient_A(fit_results, figwidth=5.5, figheight=4.5, savefile=None, saveformat='png', title="Fermi-liquid A coefficient"):
    import pandas as pd

    # Collect all coefficients from fit_results
    records = []
    for (pressure, run), result in fit_results.items():
        params = result["params"]
        Trange = result["Trange"]
        error = result["error"]

        records.append({
            "Pressure": pressure,
            "RunLabel": run,
            "A_coeff": params[1],   # A
            "rho0": params[0],      # rho₀
            "Tmin": Trange[0],
            "Tmax": Trange[1],
            "Error": error
        })

    df = pd.DataFrame(records)

    # Print coefficients line by line
    print(f"\nFermi-liquid A coefficients for {title}:")
    print(f"{'Pressure(GPa)':>14}  {'RunLabel':>10}  {'A_coeff':>12}")
    for _, row in df.iterrows():
        print(f"{row['Pressure']:>14}  {row['RunLabel']:>10}  {row['A_coeff']:>12.4e}")

    # Plot A vs Pressure, grouped by run label
    print("start plotting coeff_A vs Pressure")
    plt.figure(figsize=(figwidth, figheight))
    for pressure in df['Pressure'].unique():
        for run in df[df['Pressure'] == pressure]['RunLabel'].unique():
            subset = df[(df['Pressure'] == pressure) & (df['RunLabel'] == run)]
            plt.scatter(subset['Pressure'], subset['A_coeff'], 
                        marker='o', color = cmap(norm(pressure)), s=40, label=f'{pressure} GPa, {run}')

        # if more than one run per pressure: label=run 
        # if one run per pressure: label = f"{run} {subset['Pressure'].iloc[0]} GPa"

    plt.xlim(6.5,14.5)
    plt.ylim(2.5e-10, 1.5e-8)
    plt.xlabel("Pressure (GPa)")
    plt.ylabel(r"A from Fermi-Liquid fit ($\rho \sim \rho_0 + A \dot T^2$)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()

    if savefile:
        plt.savefig(f"{title}.{saveformat}", dpi=600, transparent=True)
    plt.show()
####################



#########
def plot_FermiLiquid_offset(fit_results, target_pressures,
                            slope_values=None, offset_values=None,
                            figwidth=10, figheight=6,
                            x='Temp1 (K)', y='resistivity',
                            savepath='FermiLiquid_T2_offset.png',
                            inset_coeff=False):

    Temp_mask = 45 # Mask for temperatures below this value

    # Setup figure
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    plt.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.1)

    # slope values default: fitted A coefficient
    if slope_values is None:
         slope_values = [result["params"][1] 
                         for (p, run_label), result in fit_results.items() if p in target_pressures]
    
    pressure_offset_slope_map = {
        p: (offset_values[i], slope_values[i]) for i, p in enumerate(target_pressures)
    }
    
    ylim_min, ylim_max = float('inf'), float('-inf')
    for (p, run_label), result in fit_results.items():
        subset = result["data"]
        params = result["params"]
        Trange = result["Trange"]

        # read slope and offset
        offset, slope = pressure_offset_slope_map.get(p, (0, 1))
        color = cmap(norm(p))
        
        subset_lowT = subset[subset['Temp1 (K)'] < Temp_mask]
        T_data = subset_lowT[x].values
        R_data = subset_lowT[y].values

        # plot original data
        ax.plot(T_data**2, R_data/slope + offset, 
                marker='o', markersize=1.0,
                linestyle='None',
                #label=f"{p} GPa", 
                alpha=0.7, color=color)

        # Plot fit curve
        T_fit = np.linspace(0, Temp_mask, 1000)
        fit_curve = Landau_resistivity(T_fit, *params)
        ax.plot(T_fit**2, fit_curve/slope + offset, 
                color='black', linestyle='-' ,lw=1.0)

       # add labels for Pressure next to each dataset
        x_text = T_fit[-1]**2
        y_text = fit_curve[-1]/slope + offset
        ax.text(
            x_text+0.02*(ax.get_xlim()[1] - ax.get_xlim()[0]),  # small offset to the right
            y_text,
            f"{p} GPa",
            va='center', ha='left', fontsize=15, color=color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1)
        )

        # Annotate Tmin and Tmax on the fit with arrows
        arrow_length = 500
        for T, color in zip(Trange, ['cyan', 'red']):
            x_arrow = T**2
            y_arrow = Landau_resistivity(T, *params)/slope + offset
            ax.annotate(
                '', 
                xy=(x_arrow, y_arrow), 
                xytext=(x_arrow, y_arrow - arrow_length), 
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0)
            )

        ### update the ylim_min, ylim_max
        yvals = fit_curve/slope + offset
        ylim_min = min(ylim_min, np.min(yvals))
        ylim_max = max(ylim_max, np.max(yvals))

    # Axis labels and limits
    ax.set_xlabel(r"$T^2$ (K$^2$)")
    ax.set_ylabel(r"Normalized, offset resistivity $\rho$ ($\Omega \cdot$m)")
    ax.text(0.05, 0.85, r"$\rho \sim \rho_0 + A \cdot T^2$",
            transform=ax.transAxes, ha='left', fontsize=15)
    ax.set_yticks([])
    ax.set_xlim(0, Temp_mask**2)
    ax.set_ylim(ylim_min*0.85, ylim_max)  # Adjusted y-limits for better visibility

    if inset_coeff:
        A_vs_pressure = []
        for p in target_pressures:
            # Find all runs for this pressure
            runs = [result for (pp, run_label), result in fit_results.items() if pp == p]
            if runs:
                # Use the first run's params[1] (A coefficient)
                A_vs_pressure.append((p, runs[0]["params"][1]))

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        pressures, A_coeffs = zip(*sorted(A_vs_pressure))
        axins = ax.inset_axes([0.62, 0.01, 0.375, 0.18])  # [x0, y0, width, height] in axes fraction
        #axins.patch.set_alpha(0.7)
        colors = [cmap(norm(p)) for p in pressures]
        axins.scatter(pressures, A_coeffs, c=colors, s=30)
        axins.set_ylabel('A coeff.', fontsize=14)
        axins.tick_params(axis='x',  which='both', bottom=False, top=False, labelbottom=False)
        axins.tick_params(axis='y', direction='in', which='both')
        axins.tick_params(labelsize=14)

    # Save and/or show
    plt.tight_layout()
    plt.savefig(savepath, dpi=600, transparent=True)
    plt.show()


########## vertical subplots
def plot_FermiLiquid_stack(original_data, fit_results,
                            target_pressures,
                            x='Temp1 (K)', y='resistivity',
                            figwidth=6, figheight_per_plot=1,
                            savepath='FermiLiquid_T2.pdf'):

    # Filter only relevant fit results
    filtered_results = [(key, result) for key, result in fit_results.items() if np.isclose(key[0], target_pressures).any()]
    n_plots = len(filtered_results)

    # Set up the figure and axes
    fig, axs = plt.subplots(n_plots, 1,
                            figsize=(figwidth, figheight_per_plot * n_plots), 
                            sharex=True, gridspec_kw={'hspace': 0})  # no vertical gap
    if n_plots == 1:
        axs = [axs]

    for ax, ((pressure, run_label), result) in zip(axs, filtered_results):
        subset_allT = result["data"]
        params = result["params"]
        Trange = result["Trange"]
        color = cmap(norm(pressure))
        
        ### Low T mask for y_lim
        subset = subset_allT[subset_allT[x] < 45]  ## mask: Temp < 45 K data
        ymin, ymax = subset[y].min(), subset[y].max()
        padding = 0.05 * (ymax - ymin) if ymax != ymin else 1e-8
        ax.set_ylim(ymin - padding, ymax + padding)
        
        # Scatter: original data
        ax.scatter(subset[x]**2, subset[y],
                   label=f"{pressure} GPa, {run_label}", #, A = {best_params[1]:.2e} ± {np.sqrt(best_covariance[1, 1]):.1e}",
                   s=1.5, alpha=0.8, color=color)

        # Line: fitted curve
        T_fit = np.linspace(0, 45, 1000)
        fit_curve = Landau_resistivity(T_fit, *params)
        ax.plot(T_fit**2, fit_curve, color='black', lw=1.0)

        # Annotate Tmin, Tmax
        ax.axvline(x=Trange[0]**2, color='cyan',
                   linestyle='-.', lw=0.5, alpha=0.8,
                   label=r'$T_{\min}$'+f': {Trange[0]} K')
        ax.axvline(x=Trange[1]**2, color='red',
                   linestyle='-.', lw=0.5, alpha=0.8,
                   label=r'$T_{\max}$'+f': {Trange[1]} K'
                   )
        
        # Label only the last subplot
        if ax != axs[-1]:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(r"$T^2$ (K$^2$)")

        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.set_yticks([])

    # Set consistent x-limits across all plots
    axs[-1].set_xlim(0, 45**2)
    fig.text(0.0, 0.5,
             r"Normalized, offset resistivity $\rho$ ($\Omega \cdot$m)",
             va='center', rotation='vertical')

    # Save and show
    plt.tight_layout()
    plt.savefig(savepath, dpi=900, transparent=True)
    plt.show()
