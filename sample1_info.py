# sample info
Project="FePSe3_Sample1"

# sample size (unit: mm)
h = 0.01*10**(-3)
w = 0.22*10**(-3) 
l = 0.23*10**(-3)

# data path
data_path='/home/sd864/work_SDeng/FePSe3/investigations/FePSe3_HP_Resistivity/Oct2023_Sample1/original_data'

# files to be loaded and analyzed
file_labels = {
    'RT_FePSe_6GPa_Tdn'       : ('6.0' , 'Tdn'),
    'RT_FePSe_7GPa_Tdn'       : ('7.0' , 'Tdn'),
#    'RP_FePSe_7GPa'           : ('7.0' , 'tbc'),
#    'RT_FePSe_7GPa_Tup_1T'    : ('7.0' , 'Tup, 1T'),
    'RT_FePSe_8GPa_Tdn'       : ('8.0' , 'Tdn'),
    'RT_FePSe_10GPa_Tdn'      : ('10.0', 'Tup'),
    'RT_FePSe_12GPa_Tdn'      : ('12.0', 'Tdn'),
    'RT_FePSe_13p5GPa_Tdn'    : ('13.5', 'Tdn'),
#    'RT_FePSe_13p5GPa_Tdn_5K' : ('13.5', 'Tdn, 5K'),
#    'RT_FePSe_13p5GPa_Tup_3T' : ('13.5', 'Tup, 3T')
}

## additional data:'RH_FePSe_13p5GPa_2K_0-5T,' : ('13.5', '2K, 0-5T'),


# check cooling rate
Pressure_check={6}  #{2,4,6,8,10,11,12,14}
Tmin=1.5
Tmax=300
fit_slope=True
Tmin_fit=2
Tmax_fit=280

### fermi_liquid_inputs
metal_pressures=[7,8,10,12,13.5]
slope_values   =[7,8.5,1.6,1,1,1]
offset_values  =[-2.73e-5,0,0,0,0,0]

