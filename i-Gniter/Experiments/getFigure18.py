import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors


def getData(file_name):
    data = pd.read_excel(file_name)
    col = data.columns.values[1:]
    data2 = {
        "index": [],
        "type": [],
        "value": []
    }
    model = data.columns.values[0]
    for i in range(len(data)):
        for j in col:
            data2["index"].append(data[model][i])
            data2["type"].append(j)
            data2["value"].append(data[j][i])
    return pd.DataFrame(data2)


def unifiedProcessingBarPlot(barplot):
    # ignore legend title
    barplot.legend(title='')
    # Bold Border
    for bar in barplot.containers:
        for rect in bar:
            rect.set_linewidth(3)
    leg = barplot.legend(frameon=True, framealpha=1, edgecolor='black',
                         handlelength=2, labelspacing=0.2, handletextpad=0.5,fontsize=22)
    leg.get_frame().set_linewidth(3.0)
    # leg.get_frame().set_boxstyle("square,pad=0.01")


def unifiedProcessingAx(ax,yticks):
    ax.set_ylabel('')
    ax.set_xlabel('')
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.yaxis.set_ticks(yticks)
    for y in yticks:
        ax.axhline(y=y, color='#C8C8C8', linestyle='-',zorder=0)
    ax.xaxis.set_tick_params(which='both', direction='in', width=3)
    ax.yaxis.set_tick_params(which='both', direction='in', width=3)

    labels = ax.get_xticklabels()
    for label in labels:
        label.set_size(20)
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_size(20)

fig = plt.figure(figsize=(12,5))
penguins = getData("Figure18.xlsx")
palette = ['#FF0000','#1400FF','#029F56','#00B2C8']

sns.set(style='ticks', context='notebook', font_scale=1.2)
barplot = sns.barplot(x="index", y="value", hue="type",
                      palette=palette, capsize=.1, data=penguins, edgecolor='black')




unifiedProcessingAx(barplot,[0,20,40,60,80,100])
unifiedProcessingBarPlot(barplot)
fig.text(0.04, 0.5, 'GPU resources (%)', ha='center', va='center', rotation='vertical',fontsize=26)
# fig.text(0.5, 0.01, 'Workloads', ha='center', va='bottom',fontsize=26)
# fig.text(0.5, 0.83, 'AlexNet', ha='center', va='top',fontsize=20,fontweight='bold')
# fig.text(0.5, 0.36, 'ResNet-50', ha='center', va='bottom',fontsize=20,fontweight='bold')

plt.subplots_adjust(left=0.1, bottom=0.10, right=0.95, top=0.9, hspace=0.2, wspace=0.2)
plt.show()
