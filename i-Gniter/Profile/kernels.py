from os import name
import sqlite3
import json
from itertools import groupby
from typing import ItemsView
import numpy as np
import matplotlib.pyplot as plt
import sys

def conse(path):
    #获取并行进程的开始和结束时间
    con = sqlite3.connect(path)
    cur = con.cursor()
    sql = "SELECT start,end,globalPid,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    cur.execute(sql)
    data=cur.fetchall()
    number = 0
    for i in data:
        number+=1
    print(number)

if __name__ == '__main__':
    ans=conse(sys.argv[1])
