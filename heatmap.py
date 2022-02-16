import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def linear_scale(old_min, old_max, new_min, new_max, old_value):
    return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


pd.set_option('display.max_columns', None)
data = pd.read_csv('heatmap.csv', sep=',', comment='#', skiprows=1)[["ppwi", "wgsize", "sum_ms"]]
# data.sort_values(by=['ppwi'], ascending=False)

normalised = data.copy()

normalised["sum_ms"] = normalised["sum_ms"].apply(
    lambda x: linear_scale(normalised["sum_ms"].min(), normalised["sum_ms"].max(), 1, 0, x))

out = normalised.pivot(index="ppwi", columns="wgsize", values="sum_ms")
out.sort_index(level=0, ascending=False, inplace=True)

# data = np.genfromtxt('heatmap.csv', delimiter=',')
print(out)

sns.heatmap(out, annot=True)

plt.show()
