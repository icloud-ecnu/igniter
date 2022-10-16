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
    sql = "SELECT start,end,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL"
    cur.execute(sql)
    data=cur.fetchall()
    conse = {}
    #key = pid, value = start,end
    for i in data:
        if(i[2] not in conse):
            conse[i[2]]={}
            conse[i[2]]["start"]=i[0]
            conse[i[2]]["end"]=i[1]
        else:
            if(conse[i[2]]["end"]<i[1]):
                conse[i[2]]["end"]=i[1]
    ansstart = 0
    ansend = sys.maxsize
    for key in conse:
        if(conse[key]["start"]>ansstart):
            ansstart = conse[key]["start"]
        if(conse[key]["end"]<ansend):
            ansend = conse[key]["end"]
    con.close()
    if(ansstart+1000000000>ansend):
        print("duration time is too short!")
    return ansstart+300000000,ansend-100000000

def sepper(path,ts,te):
    # 间隙时间
    con = sqlite3.connect(path)
    cur = con.cursor()
    sql = "SELECT start,end,globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start > "+ ts +" AND end < "+ te
    cur.execute(sql)
    data=cur.fetchall()
    sep = {}
    #key = pid, value = data
    for i in data:
        if(i[2] not in sep):
            sep[i[2]]={}
            sep[i[2]]["totals"]=i[0]
            sep[i[2]]["totale"]=i[1]
            sep[i[2]]["tmp"]=i[1]
            sep[i[2]]["sep"]=0
            #print(sep[i[2]]["totals"])
        else:
            sep[i[2]]["totale"]=i[1]
            sep[i[2]]["sep"]+=(i[0]-sep[i[2]]["tmp"])
            sep[i[2]]["tmp"]=i[1]
    anssep=[]
    for key in sep:
        #print(sep[key]["sep"],sep[key]["totale"],sep[key]["totals"])
    #    print(sep[key]["sep"]/(sep[key]["totale"]-sep[key]["totals"]))
        anssep.append(sep[key]["sep"]/(sep[key]["totale"]-sep[key]["totals"]))
    con.close()
    return anssep

def runtime(path,ts,te,instance):
    #每个推理运行时间
    con = sqlite3.connect(path)
    cur = con.cursor()
    sql = "SELECT start,end,globalPid,value FROM CUPTI_ACTIVITY_KIND_KERNEL LEFT OUTER JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.demangledName = StringIds.id  WHERE start > "+ ts +" AND end < "+ te
    # [0] strart, [1] end, [2] globalPid, [3] value
    cur.execute(sql)
    data=cur.fetchall()
    avginf = {}
    #key = pid, value = data 
    for i in data:
        if(i[2] not in avginf):
            avginf[i[2]]={}
            avginf[i[2]]["dur"]=[]
            avginf[i[2]]["p"]=1
            avginf[i[2]]["ts"]=i[0]
            avginf[i[2]]["fname"]=i[3]
        else:
            if(avginf[i[2]]["p"]==int(instance)):
                avginf[i[2]]["dur"].append(i[1]-avginf[i[2]]["ts"])
                avginf[i[2]]["p"]=0
            if(avginf[i[2]]["p"]==1):
                if(i[3]!=avginf[i[2]]["fname"]):
                    print("name warning")
                    print(avginf[i[2]]["fname"])
                    print(i[3])
                    return
                    avginf[i[2]]["fname"]=i[3]
                avginf[i[2]]["ts"]=i[0]
        avginf[i[2]]["p"] += 1
    anssep=[]        
    for pid in avginf:
     #   print(np.mean(avginf[pid]["dur"])/1000/1000)
        #print(np.mean(avginf[pid]["dur"])/1000/1000,np.max(avginf[pid]["dur"])/1000/1000,len(avginf[pid]["dur"]))
        anssep.append(np.mean(avginf[pid]["dur"])/1000/1000)
    con.close()
    return anssep

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
    data=0
    j=1
    while(stream==data[j][0]):
        j+=1
    for i in data[j:-1]:
        if(stream==i[0]):
            data+=i[1]
            flag=1
        else:
            if(flag==0):
                data+=i[1]
            else:
                break
    con.close()
    
    return data



if __name__ == '__main__':
    ans=conse(sys.argv[1])
    #print(ans[0],ans[1])
    sep=sepper(sys.argv[1],str(ans[0]),str(ans[1]))
    run=runtime(sys.argv[1],str(ans[0]),str(ans[1]),sys.argv[2])
    idle=[]
    for i in range(len(sep)):
        idle.append(sep[i]*run[i])
    print(np.mean(idle))

