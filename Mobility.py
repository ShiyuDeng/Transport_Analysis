# analyze the mobility of charge carriers

def mobility_analysis(Hall_eff_data='Hall_effect_data',
                      resistivity_data='sample1_rho0',
                      figwidth=6.0, figheight=4.0,
                      saveplot=None, saveformat='png'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    """
    Calculate the mobility of charge carriers given resistivity rho and carrier density n
    conductivity \sigma = n e \mu
    \sigma = 1 / \rho

    n is obtained from Hall effect measurements:
    RH = 1/(n e)

    Hall_eff_data is from an external file containing Hall effect measurements.
    resistivity_data is a DataFrame containing resistivity measurements, 
                          from FermiLiquid fitting parameters,
                          containing columns: 'Pressure', 'rho0'
    """
    e = 1.602176634e-19  # electron charge constant in C

    ###############  R_H ##################
    # RH is mesured as a function of pressure, reading from a data file
    df_hall= pd.read_csv(Hall_eff_data, sep=r'\s+')
    df_hall.columns = ['Pressure', 'R_H']
    df_hall['R_H'] = df_hall['R_H']*10**(-9)
    df_hall['n'] = 1 / (df_hall['R_H'] * e)  # carrier density n in m^-3
    
    #debug
    print(df_hall)
    plt.figure(figsize=(figwidth, figheight))
    plt.axvspan(7, 13.5, color='lightgray', alpha=0.5, zorder=-10, 
                label='Pressure Range During Our measurements')
    
    plt.plot(df_hall['Pressure'], df_hall['n'],
             marker='^', linestyle=':', color='orange', linewidth=0.1)

    # Polynomial or spline fit for n(P)
    fitted_n = interp1d(df_hall['Pressure'], df_hall['n'],
                        kind='linear', bounds_error=False, fill_value='extrapolate')
    
    pressures = np.linspace(df_hall['Pressure'].min(), df_hall['Pressure'].max(), 1000)
    n_values = fitted_n(pressures)
    plt.plot(pressures, n_values, color='orange', linestyle='-.', linewidth=1.0,
             label='Fitted n(P)', zorder=-1)
    plt.xlabel('Pressure (GPa)')
    plt.ylabel(r'Carrier Density $n$ (m$^{-3}$)', color='orange')

    # add twin y-axis for R_H
    ax2 = plt.gca().twinx()
    ax2.plot(df_hall['Pressure'], df_hall['R_H'],
             marker='o', linestyle='-.', color='green', linewidth=1.0, label='R_H')
    ax2.set_ylabel(r'Hall Coefficient $R_H$ (m$^3$/C)', color='green')
    ax2.text(0.25, 0.5, r'$R_H$ measured at 10~K with PPMS-9', 
             color='green', transform=ax2.transAxes, fontsize=14)
    ax2.text(0.25, 0.3, r'n = 1 / ($R_H$ *e )', 
            color='orange', transform=ax2.transAxes, fontsize=14)
    plt.grid(True, linestyle='-.', linewidth=0.5)
    plt.title('Data extracted from Wang et al. 2018')

    ##################  \sigma (1/\rho) ###########################
    # Read resistivity data
    # how to ensure the same accuracy as the data file
    df_resistivity = pd.read_csv(resistivity_data, sep=r'\s+')
    df_resistivity.columns = ['Pressure', 'RunLabel', 'rho0']
    pd.set_option('display.precision', 6)
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(df_resistivity.to_string(index=False))

    # Estimate mobility at each pressure
    mobility_data = []
    for _, row in df_resistivity.iterrows():
        pressure = row['Pressure']
        rho = row['rho0']
        n_interp = fitted_n(pressure)

        if n_interp > 0 and rho > 0:
            mu = 1 / (rho * n_interp * e)  # m²/(V·s)
        else:
            mu = np.nan  # handle invalid values

        mobility_data.append({
            'Pressure': pressure,
            'rho0': rho,
            'mobility': mu
        })


    df_mu = pd.DataFrame(mobility_data).sort_values('Pressure')
    print(df_mu)

    # Plotting mobility vs Pressure
    plt.figure(figsize=(figwidth, figheight))
    plt.plot(df_mu['Pressure'], df_mu['mobility'],
             marker='H', linestyle='-.', color='blue', linewidth=0.5)
    plt.xlabel('Pressure (GPa)')
    plt.ylabel(r'Mobility $\mu = 1/(\rho * n *e)$ (m$^2$/V·s)')
    plt.title('Sample1: Mobility vs Pressure')

    # add twin plot of n(Pressure)
    ax3 = plt.gca().twinx()
    ax3.plot(df_hall['Pressure'], df_hall['n'],
             marker='^', linestyle=':', color='orange', linewidth=0.5, label='n(P)')
    ax3.set_ylabel(r'Carrier Density $n$ (m$^{-3}$)', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    ax3.text(0.55, 0.2, r'n = 1 / ($R_H$ *e )', 
             color='orange', transform=ax3.transAxes, fontsize=14)
    
    annotation = (
        r'$\mu = \frac{1}{\rho_0 \cdot n \cdot e}$' + '\n' +
        r'$\rho_0$: from our low T fit' + '\n' +
        r'$n$: from Wang et al., $R_H$'
    )
    plt.text(0.05, 0.7, annotation, transform=plt.gca().transAxes, color='blue',fontsize=14)
    plt.grid(True, linestyle='-.', linewidth=0.5)
    ax3.set_xlim(7, 14)
    plt.tight_layout()

    if saveplot:
        plt.savefig(f"{saveplot}.{saveformat}", dpi=600, transparent=True)
    plt.show()


    ## plot mu as a function of pressure
        
    return df_resistivity