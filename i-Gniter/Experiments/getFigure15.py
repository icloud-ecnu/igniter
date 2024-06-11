import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colors


def getDataArray(file_name):
    data = pd.read_excel(file_name)
    print(data)
    col = data.columns.values[1:]
    ans = [[[], []], [[], []],[[], []]]
    model = data.columns.values[0]
    for j in range(len(col)):
        for i in range(len(data)):
            print(i)
            ans[j][0].append(data[model][i])
            ans[j][1].append(data[col[j]][i])
    print(ans)
    return ans

penguinsTop = getDataArray("Figure15Top.xlsx")
penguinsDown = getDataArray("Figure15Down.xlsx")
# 设置画布大小
fig = plt.figure(figsize=(6, 4))
# 第一个图
plt.subplot(2,1,1)
plt.plot(penguinsTop[0][0], penguinsTop[0][1], '-', alpha=1, linewidth=2, label='SLO', color='#FF0000',zorder=0)  # '
plt.plot(penguinsTop[1][0], penguinsTop[1][1], '--', alpha=1, linewidth=2, label='GSLICE+', color='#00AB59',zorder=1)
plt.plot(penguinsTop[2][0], penguinsTop[2][1], '--', alpha=1, linewidth=2, label='Gniter', color='#0000CD',zorder=15)
leg = plt.legend(bbox_to_anchor=(0.11, 0.25), loc='upper left', borderaxespad=0.,ncol=3,edgecolor='black',fontsize=14)
leg.get_frame().set_linewidth(2.0)
plt.ylabel('Throughput (req/s)', fontsize=12, labelpad=4)
# 调整刻度线方向
plt.tick_params(which='both', direction='in', width=1)
# 设置图边框粗细
ax = plt.gca()
[i.set_linewidth(2) for i in ax.spines.values()]
# 设置y轴范围
plt.ylim(95, 155)
plt.yticks(np.arange(100,151,10))
# 设置x轴范围
plt.xlim(0,63)
plt.xticks(np.arange(0,61,5))
# 画一个红色的圆圈
plt.plot(61, 150, 'ro', markersize=10, linewidth=10, markerfacecolor='none', markeredgecolor='red',zorder=5)


# 第二个图
plt.subplot(2,1,2)
plt.plot(penguinsDown[0][0], penguinsDown[0][1], '--', alpha=1, linewidth=2, label='1/2 SLO', color='#FF0000')  # 0000CD
plt.plot(penguinsDown[1][0], penguinsDown[1][1], '--', alpha=1, linewidth=2, label='GSLICE+', color='#00AB59')#00AB59
plt.plot(penguinsDown[2][0], penguinsDown[2][1], '-', alpha=1, linewidth=2, label='iGniter', color='#0000CD')#FF0000
leg = plt.legend(bbox_to_anchor=(0.3, 0.99), loc='upper left', borderaxespad=0.,ncol=2,edgecolor='black',fontsize=14,handletextpad=0.01,labelspacing=0.2)
leg.get_frame().set_linewidth(2.0)
plt.ylabel('Latency (ms)', fontsize=12, labelpad=4)
plt.xlabel('Time (seconds)', fontsize=12, labelpad=4)  # accuracy
# 调整刻度线方向
plt.tick_params(which='both', direction='in', width=1)
# 设置图边框粗细
ax = plt.gca()
[i.set_linewidth(2) for i in ax.spines.values()]
# 设置y轴范围
plt.ylim(4,20)
plt.yticks(np.arange(8,21,4))
# 设置x轴范围
plt.xlim(0,63)
plt.xticks(np.arange(0,61,5))
# 画一个红色的圆圈
plt.plot(61, 12.5, 'ro', markersize=10, linewidth=10, markerfacecolor='none', markeredgecolor='red',zorder=5)





# 调整留白
plt.subplots_adjust(left=0.12, bottom=0.14, right=0.95, top=0.9, hspace=0.2, wspace=0.2)

plt.show()