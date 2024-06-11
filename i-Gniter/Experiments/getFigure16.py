import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import colors


def getDataArray(file_name):
    data = pd.read_excel(file_name)
    print(data)
    col = data.columns.values[1:]
    ans = [[[], []], [[], []]]
    model = data.columns.values[0]
    for j in range(len(col)):
        for i in range(len(data)):
            ans[j][0].append(data[model][i])
            ans[j][1].append(data[col[j]][i])
    print(ans)
    return ans

penguinsTop = getDataArray("Figure16Top.xlsx")
penguinsDown = getDataArray("Figure16Down.xlsx")
# 设置画布大小
fig = plt.figure(figsize=(6, 4))
# 第一个图
plt.subplot(2,1,1)
plt.plot(penguinsTop[0][0], penguinsTop[0][1], '--', alpha=1, linewidth=2, label='GSLICE+', color='#00AB59',zorder=0)
plt.plot(penguinsTop[1][0], penguinsTop[1][1], '-', alpha=1, linewidth=2, label='iGniter', color='#0000CD',zorder=1)
leg = plt.legend(bbox_to_anchor=(0.35, 0.42), loc='upper left', borderaxespad=0.,ncol=1,edgecolor='black',fontsize=16,labelspacing=0.2)
leg.get_frame().set_linewidth(2.0)
plt.ylabel('Batch size', fontsize=12, labelpad=4)
# 调整刻度线方向
plt.tick_params(which='both', direction='in', width=1)
# 设置图边框粗细
ax = plt.gca()
[i.set_linewidth(2) for i in ax.spines.values()]
# 设置y轴范围
plt.ylim(0, 2.2)
plt.yticks(np.arange(1,3))
# 设置x轴范围
plt.xlim(0,65)
plt.xticks(np.arange(0,61,5))
circle = plt.Circle((0, 0), radius=1, color='red',fill=True)
# 画一个红色的圆圈
plt.plot(61, 2, 'ro', markersize=10, linewidth=10, markerfacecolor='none', markeredgecolor='red',zorder=5)


# 第二个图
plt.subplot(2,1,2)
plt.plot(penguinsDown[0][0], penguinsDown[0][1], '--', alpha=1, linewidth=2, label='GSLICE+', color='#00AB59')  # 0000CD
plt.plot(penguinsDown[1][0], penguinsDown[1][1], '-', alpha=1, linewidth=2, label='iGniter', color='#0000CD')#00AB59
leg = plt.legend(bbox_to_anchor=(0.685, 0.4), loc='upper left', borderaxespad=0.,ncol=1,edgecolor='black',fontsize=14,labelspacing=0.2)
leg.get_frame().set_linewidth(2.0)
plt.ylabel('GPU resources (%)', fontsize=12, labelpad=4)
plt.xlabel('Time (seconds)', fontsize=12, labelpad=4)  # accuracy
# 调整刻度线方向
plt.tick_params(which='both', direction='in', width=1)
# 设置图边框粗细
ax = plt.gca()
[i.set_linewidth(2) for i in ax.spines.values()]
# 设置y轴范围
plt.ylim(25,80)
plt.yticks(np.arange(30,81,10))
# 设置x轴范围
plt.xlim(0,65)
plt.xticks(np.arange(0,61,5))
# 画一个红色的圆圈
plt.plot(61, 75, 'ro', markersize=10, linewidth=10, markerfacecolor='none', markeredgecolor='red',zorder=5)



# 调整留白
plt.subplots_adjust(left=0.12, bottom=0.14, right=0.95, top=0.9, hspace=0.2, wspace=0.2)

plt.show()