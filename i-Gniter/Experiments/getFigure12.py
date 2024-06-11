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
    leg = barplot.legend(frameon=True, framealpha=1, edgecolor='black',handlelength=2, labelspacing=0.2, handletextpad=0.5)
    leg.get_frame().set_linewidth(2.0)
    # leg.get_frame().set_boxstyle("square,pad=0.01")


def unifiedProcessingAx(ax,yticks):
    ax.set_ylabel('')
    ax.set_xlabel('')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.yaxis.set_ticks(yticks)
    for y in yticks:
        ax.axhline(y=y, color='#C8C8C8', linestyle='-',zorder=0)
    ax.xaxis.set_tick_params(which='both', direction='in', width=3)
    ax.yaxis.set_tick_params(which='both', direction='in', width=3)

    labels = ax.get_xticklabels()
    for label in labels:
        label.set_size(16)
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_size(16)


penguinsResNet = getData("Figure12_ResNet.xlsx")
penguinsAlexNet = getData("Figure12_AlexNet.xlsx")


palette = ['#FF0000','#1400FF','#00AB59']
fig, (axAlexNet,axResNet) = plt.subplots(2, 1, figsize=(6, 4))


sns.set(style='ticks', context='notebook', font_scale=1.2)
barplotAlexNet = sns.barplot(x="index", y="value", hue="type",
                      palette=palette, capsize=.1, ax=axAlexNet, data=penguinsAlexNet, edgecolor='black',dodge=0.2)

barplotResNet = sns.barplot(x="index", y="value", hue="type",
                      palette=palette, capsize=.1, ax=axResNet, data=penguinsResNet, edgecolor='black')



unifiedProcessingAx(axAlexNet,[0,2,4,6,8])
unifiedProcessingAx(axResNet,[0,10,20,30,40])
unifiedProcessingBarPlot(barplotAlexNet)
unifiedProcessingBarPlot(barplotResNet)
fig.text(0.06, 0.5, 'Inference latency (ms)', ha='center', va='center', rotation='vertical',fontsize=18)
fig.text(0.5, -0.01, 'Batch size', ha='center', va='bottom',fontsize=18)
fig.text(0.5, 0.83, 'AlexNet', ha='center', va='top',fontsize=20,fontweight='bold')
fig.text(0.5, 0.36, 'ResNet-50ResNet-50', ha='center', va='bottom',fontsize=20,fontweight='bold')

plt.show()
