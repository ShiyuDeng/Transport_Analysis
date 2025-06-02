import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["green", "orange", "blue", "red"])
norm = plt.Normalize(2, 14)
markers = ['o', '^', '+', 'p','v', 'x', '*', 's', 'D']
