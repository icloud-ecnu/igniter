import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors, ticker
from matplotlib.patches import FancyArrow


def getDataArray(file_name):
    data = pd.read_excel(file_name)
    col = data.columns.values[1:]
    ans = [[[], []], [[], []], [[], []], [[], []]]
    model = data.columns.values[0]
    for j in range(len(col)):
        for i in range(len(data)):
            ans[j][0].append(data[model][i])
            ans[j][1].append(data[col[j]][i])
    print(ans)
    return ans


def unifiedProcessingAx(ax, yticks,color):


    ax.yaxis.set_tick_params(which='both', direction='in', width=1)
    ax.xaxis.set_tick_params(which='both', direction='in', width=1)
    ax.set_ylabel('')
    ax.set_xlabel('')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.yaxis.set_ticks(yticks)
    ax.xaxis.set_tick_params(which='both', direction='in', width=1)
    ax.yaxis.set_tick_params(which='both', direction='in', width=1,color=color)


    labels = ax.get_xticklabels()
    for label in labels:
        label.set_size(14)
    labels = ax.get_yticklabels()
    for label in labels:
        label.set_size(14)
        label.set_color(color)



penguinsScatter = getDataArray("Figure21Scatter.xlsx")
penguinsLine = getDataArray("Figure21Line.xlsx")
# 创建一个共享 x 轴的双 y 轴图
fig, ax1 = plt.subplots(figsize=(6, 4))
ax2 = ax1.twinx()
plt.xscale('log')
plt.xlim(9, 1002)
plt.xticks([10, 100, 1000])

ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
# 左边y轴绘图
ax2.spines['left'].set_color('#FF0000')
ax1.set_yscale('log')
ax1.set_ylim(0.001, 10)
unifiedProcessingAx(ax1, [0.001,0.01,0.1,1,10],'r')
ax1.errorbar(penguinsScatter[0][0], penguinsScatter[0][1], yerr=penguinsScatter[1][1], fmt='x', ecolor='#FF0000',
            color='#FF0000', elinewidth=1, capsize=4)
ax1.plot(penguinsLine[0][0], penguinsLine[0][1], '-', alpha=1, linewidth=1, color='#FF0000')  # '

# 右边y轴绘图
ax2.spines['right'].set_color('#0000CD')
ax2.set_ylim(52.5, 55)
unifiedProcessingAx(ax2, np.arange(53, 56, 1),'b')
ax2.errorbar(penguinsScatter[2][0], penguinsScatter[2][1], markerfacecolor='none',yerr=penguinsScatter[3][1], fmt='o', ecolor='#0000CD',
            color='#0000CD', elinewidth=1, capsize=4)
ax2.plot(penguinsLine[1][0], penguinsLine[1][1], '--', alpha=1, linewidth=1, color='#0000CD')  # '




fig.text(0.06, 0.5, 'Computation overhead (s)', ha='center', va='center', rotation='vertical',fontsize=17,color='#FF0000')
fig.text(0.94, 0.5, 'Memory overhead (MB)', ha='center', va='center', rotation='vertical',fontsize=17,color='#0000CD')
fig.text(0.5, 0.02, 'Number of workloads', ha='center', va='bottom',fontsize=18)
# 调整留白
plt.subplots_adjust(left=0.18, bottom=0.16, right=0.84, top=0.9, hspace=0.2, wspace=0.2)

plt.show()
