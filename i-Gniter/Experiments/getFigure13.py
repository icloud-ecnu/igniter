import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors


def getData(file_name):
    data = pd.read_excel(file_name)
    col = data.columns.values[1:-1]
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
    # 设置图例
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

fig = plt.figure(figsize=(8,6))
penguins = getData("Figure13.xlsx")
palette = ['#FF0000','#00AB59']

sns.set(style='ticks', context='notebook', font_scale=1.2)
barplot = sns.barplot(x="index", y="value", hue="type",
                      palette=palette, capsize=.1, data=penguins, edgecolor='black')




unifiedProcessingAx(barplot,[0,5,10,15,20,25,30,35])
unifiedProcessingBarPlot(barplot)
fig.text(0.05, 0.5, 'Inference latency (ms)', ha='center', va='center', rotation='vertical',fontsize=26)
fig.text(0.5, 0.015, 'Workloads', ha='center', va='bottom',fontsize=26)
# fig.text(0.5, 0.83, 'AlexNet', ha='center', va='top',fontsize=20,fontweight='bold')
# fig.text(0.5, 0.36, 'ResNet-50', ha='center', va='bottom',fontsize=20,fontweight='bold')

plt.subplots_adjust(left=0.13, bottom=0.16, right=0.9, top=0.9, hspace=0.2, wspace=0.2)
plt.show()
