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
    sql = "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    cur.execute(sql)
    data=cur.fetchall()
    ansend=data[-1][0]-1000000000
    ansstart=ansend-1000000000
    return ansstart,ansend



def transferdata(path,ts,te):
    # 间隙时间 
    # 统计两个streaam的和
    con = sqlite3.connect(path)
    cur = con.cursor()
    sql = "SELECT streamId,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start > "+ ts +" AND end < "+ te
    cur.execute(sql)
    data=cur.fetchall()
    stream = data[0][0]
    flag = 0
    transferdata=0
    j=1
    #print(stream, data[0][0])
    while(stream==data[j][0]):
        j+=1
    for i in data[j:-1]:
        if(stream==i[0]):
            transferdata+=i[1]
            flag=1
        else:
            if(flag==0):
                transferdata+=i[1]
            else:
                break
    con.close()
    return transferdata

def saverecords(kind,metrics,value):
    path="config"
    records={kind:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")


if __name__ == '__main__':
    ans=conse(sys.argv[1])
    #print(ans[0],ans[1])
    transferdata=transferdata(sys.argv[1],str(ans[0]),str(ans[1]))
    saverecords(sys.argv[2], "transferdata",transferdata)
    print(transferdata)