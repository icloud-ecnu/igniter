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

penguinsSSD = getData("Figure11_SSD.xlsx")
penguinsVGG = getData("Figure11_VGG.xlsx")


palette = ['#FF0000','#1400FF','#00AB59']
fig, (axVGG, axSSD) = plt.subplots(2, 1, figsize=(6, 4))


sns.set(style='ticks', context='notebook', font_scale=1.2)
axVGG.set_ylim(0, 32)
barplotVGG = sns.barplot(x="index", y="value", hue="type",
                      palette=palette, capsize=.1, ax=axVGG, data=penguinsVGG, edgecolor='black',dodge=0.2)

axSSD.set_ylim(0, 49)
barplotSSD = sns.barplot(x="index", y="value", hue="type",
                      palette=palette, capsize=.1, ax=axSSD, data=penguinsSSD, edgecolor='black',order=[80,60,40,20])



unifiedProcessingAx(axVGG,[0,10,20,30])
unifiedProcessingAx(axSSD,[0,15,30,45])
unifiedProcessingBarPlot(barplotVGG)
unifiedProcessingBarPlot(barplotSSD)
fig.text(0.04, 0.5, 'Inference latency (ms)', ha='center', va='center', rotation='vertical',fontsize=18)
fig.text(0.5, 0.011, 'GPU resources (%)', ha='center', va='bottom',fontsize=18)
fig.text(0.5, 0.85, 'VGG-19', ha='center', va='top',fontsize=20)
fig.text(0.5, 0.38, 'SSD', ha='center', va='bottom',fontsize=20)


# 调整留白
plt.subplots_adjust(left=0.13, bottom=0.15, right=0.95, top=0.9, hspace=0.2, wspace=0.2)

plt.show()
