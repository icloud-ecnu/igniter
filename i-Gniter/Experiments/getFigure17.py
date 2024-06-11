import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
from matplotlib.patches import FancyArrow


def getDataArray(file_name):
    data = pd.read_excel(file_name)
    col = data.columns.values[1:]
    ans = [[[], []], [[], []]]
    model = data.columns.values[0]
    for j in range(len(col)):
        for i in range(len(data)):
            ans[j][0].append(data[model][i])
            ans[j][1].append(data[col[j]][i])
    print(ans)
    return ans


penguins = getDataArray("Figure17.xlsx")

fig = plt.figure(figsize=(6, 4))
# 画两根折线
plt.plot(penguins[0][0], penguins[0][1], '-', alpha=1, linewidth=3, label='P99 latency', color='#0000CD')  # '
plt.plot(penguins[1][0], penguins[1][1], '--', alpha=1, linewidth=3, label='SLO', color='#00AB59')
# 显示图例，即上面的label
plt.legend()
plt.ylabel('P99 latency (ms)', fontsize=16, labelpad=4)
plt.xlabel('Time (seconds)', fontsize=16, labelpad=4)  # accuracy
# 设置x轴的范围
plt.xlim(0, 20)
# 设置y轴的范围
plt.ylim(7,16.5)

# 设置图例
leg = plt.legend(frameon=True, framealpha=1, edgecolor='black',
                 handlelength=3, labelspacing=0.2, handletextpad=0.5, fontsize=14)
leg.get_frame().set_linewidth(3.0)
# 设置y轴范围
yticks = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
plt.yticks(yticks)
# 设置x轴范围
xticks = [0, 5, 10, 15, 20]
plt.xticks(xticks)
# 调整x轴y轴的刻度数字大小
plt.tick_params(axis='both', which='major', labelsize=14)
# 设置图边框粗细
ax = plt.gca()
[i.set_linewidth(2) for i in ax.spines.values()]
# plt.gca().spines['top'].set_linewidth(2)
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.gca().spines['left'].set_linewidth(2)
# plt.gca().spines['right'].set_linewidth(2)
# 设置y横线
for y in yticks:
    plt.axhline(y=y, color='#C8C8C8', linestyle='-', zorder=0)
# 调整刻度线方向
plt.tick_params(which='both', direction='in', width=1)
# 加一条垂直于x轴的校准线
plt.vlines(x=1.5, ymin=7.5, ymax=16, color='r', linestyle='--',linewidth=2)
# 在 (3, 12.5) 的位置加一个箭头，dx 是指箭头在 x 轴上的移动距离，dy 是指箭头在 y 轴上的移动距离。
arrow = FancyArrow(3, 12.5, -1, 0, width=0.1, color='red')
plt.gca().add_patch(arrow)
# 在箭头后面加文字
plt.text(3.6, 12.5, 'Switching to the',fontsize=20,color='red')
plt.text(3.6, 11.5, 'shadow process',fontsize=20,color='red')
# 调整留白
plt.subplots_adjust(left=0.12, bottom=0.14, right=0.95, top=0.9, hspace=0.2, wspace=0.2)
# 绘图
plt.show()
