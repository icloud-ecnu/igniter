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
        #print(sep[key]["sep"]/(sep[key]["totale"]-sep[key]["totals"]))
        anssep.append(sep[key]["sep"]/(sep[key]["totale"]-sep[key]["totals"]))
    con.close()
    return anssep

def runtime(path,ts,te,instance):
    #每个推理运行时间
    con = sqlite3.connect(path)
    cur = con.cursor()
    sql = "SELECT start,end,globalPid,value FROM CUPTI_ACTIVITY_KIND_KERNEL LEFT OUTER JOIN StringIds ON CUPTI_ACTIVITY_KIND_KERNEL.demangledName = StringIds.id  WHERE start > "+ ts +" AND end < "+ te
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
        #print(np.mean(avginf[pid]["dur"])/1000/1000)
        #print(np.mean(avginf[pid]["dur"])/1000/1000,np.max(avginf[pid]["dur"])/1000/1000,len(avginf[pid]["dur"]))
        anssep.append(np.mean(avginf[pid]["dur"])/1000/1000)
    con.close()
    return anssep

def multimean(a,b):
    ans=[]
    for i in range(len(a)):
        ans.append(a[i]*b[i])
    return np.mean(ans)

def saverecords(kind,metrics,value):
    path="config"
    records={kind:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")

def sysprocess():
    values=[0] * 5
    pathb="data/v100_sysidletime_"
    for i in range(1,6):
        path=pathb+str(i)+".sqlite"
        ans=conse(path)
        idlep=sepper(path,str(ans[0]),str(ans[1]))
        dur=runtime(path,str(ans[0]),str(ans[1]),29)
        values[i-1]=multimean(idlep,dur)/29
    print(values)
    saverecords("hardware","idletime",values)

def soloinference(path,model,kernels):
    ans=conse(path)
    idlep=sepper(path,str(ans[0]),str(ans[1]))
    dur=runtime(path,str(ans[0]),str(ans[1]),kernels)
    idletime=multimean(idlep,dur)
    activetime=dur[0]-idletime
    print(idletime,activetime)
    saverecords(model,"idletime_1",idletime)
    saverecords(model,"activetime_1",activetime)

def multiinference(path,model,kernels):
    ans=conse(path)
    idlep=sepper(path,str(ans[0]),str(ans[1]))
    dur=runtime(path,str(ans[0]),str(ans[1]),kernels)
    idletime=multimean(idlep,dur)
    activetime=dur[0]-idletime
    print(activetime)
    saverecords(model,"activetime_2",activetime)

def loadprofile(path):
    profile = {}
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line)
            for key in data:
                profile[key] = data[key]

    return profile


def initcontext():
    return loadprofile("./kernelConfig")

if __name__ == '__main__':
    #ans=conse(sys.argv[1])
    #print(ans[0],ans[1])
    kernels = initcontext()
    if(sys.argv[1]=="sys"):
        sysprocess()
    elif(sys.argv[3]=="1"):
        soloinference(sys.argv[1],sys.argv[2],kernels[sys.argv[2]])
    elif(sys.argv[3]=="2"):
        multiinference(sys.argv[1],sys.argv[2],kernels[sys.argv[2]])
    #idle=[]
    #for i in range(len(sep)):
    #    idle.append(sep[i]*run[i])
    #print(np.mean(idle))

