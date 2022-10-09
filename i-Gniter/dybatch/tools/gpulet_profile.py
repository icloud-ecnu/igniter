import sys
import subprocess
import soloduration
import datetime

import os
import trtexecps
import json

from numpy import DataSource
import pandas as pd
import copy
import math
import numpy as np



models=["alexnet","resnet50","vgg19","ssd"]

#models=["alexnet","ssd"]

def ncompute(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        id=0
        reader = pd.read_csv(f)
        #print(reader)
        data={}
        for i in range(0, len(reader)):
            id=reader.iloc[i]['ID']
            mn=reader.iloc[i]['Metric Name']
            if id not in data:
                data[id]={}
            #print("idmn",id,mn)
            #print("reader",reader.iloc[i]['Metric Value'])
            data[id][mn]=reader.iloc[i]['Metric Value']
        size=len(data)
        kernels=size/10
        metrics=['lts__t_sectors.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed']
        ans=[]
        tmp=[0]*len(metrics)
        totaldurationtime=0
        for i in range(size):
            if(i%kernels==0 and i>1):
                #print(totaldurationtime)
                tmp=[t/totaldurationtime for t in tmp]
                ans.append(copy.deepcopy(tmp))
                totaldurationtime=0
                for j in range(len(tmp)):
                    tmp[j]=0
            totaldurationtime+=data[i]['gpu__time_duration.sum']/1000
            for j in range(len(metrics)):
                tmp[j]+=data[i][metrics[j]]*data[i]['gpu__time_duration.sum']/1000
        tmp=[t/totaldurationtime for t in tmp]
        ans.append(copy.deepcopy(tmp))
        avgans=[0] * len(metrics)
        for i in range(len(ans)):
            for j in range(len(metrics)):
                avgans[j]+=ans[i][j]
        avgans=[i/10 for i in avgans]
        print(avgans)
        print(size)
        return avgans

def saverecords(type,metrics,value):
    path="testcon.log"
    records={type:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")

def soloduration(model):
    batches=[i for i in range(1,33)]
    #batches=[i for i in range(1,9)]
    threads=[20,40,50,60,80]
    #mpsid=os.getenv('MPSID')
    throughput=[]
    inference_duration=[]
    ans=[]
    #[batch,thread,th,lat]
    #print(mpsid)
    dram={}
    l2cache={}
    latency={}
    throughput={}
    k=0
    for i in batches:
        dram[i]={}
        l2cache[i]={}
        latency[i]={}
        throughput[i]={}        
        for j in threads:
            dram[i][j]=0
            l2cache[i][j]=0
            latency[i][j]=0
            throughput[i][j]=0
    for i in batches:
        #throughput.append([])
        #inference_duration.append([])
        for j in threads:
            subprocess.run("./soloduration_t5.sh "+model+" "+str(j)+" "+str(i), shell=True)
            outputpath="data/durtime_"+model+"_b"+str(i)+"_t"+str(j)
            dur=trtexecps.trtexecps_th(outputpath)
            # dram l2cache
            subprocess.run("./dram_l2.sh "+model+" "+str(j)+" "+str(i), shell=True)
            output="data/"+model+"_l2cache_t"+str(j)+"_b"+str(i)+".csv"
            dram_l2cache=ncompute(output)
            l2cache[i][j]=dram_l2cache[0]
            dram[i][j]=dram_l2cache[1]
            latency[i][j]=dur[1]
            throughput[i][j]=dur[0]*i
            #throughput[k].append(dur[0])
            #inference_duration[k].append(dur[1])
        k+=1
    threads2=[10,100]

    for j in threads2:
        subprocess.run("./soloduration_t5 "+model+" "+str(j)+" 1", shell=True)
        outputpath="data/durtime_"+model+"_b1_t"+str(j)
        dur=trtexecps.trtexecps_th(outputpath)
        #throughput[k].append(dur[0])
        #inference_duration[k].append(dur[1])
        latency[1][j]=dur[1]
        throughput[1][j]=dur[0]*i

    saverecords(model,"latency",latency)
    saverecords(model,"throughput",throughput)
    saverecords(model,"l2cache",l2cache)
    saverecords(model,"dram",dram)

def durationps(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        log=f.readlines()
        inferencelatency=[]
        for i in range(1, len(log)-1):
            j=eval(log[i][1:])
            #j = json.loads(log[i])
            if(j["startComputeMs"]>5000 and j["endComputeMs"]<10000):
                inferencelatency.append(j["latencyMs"])
        if(len(inferencelatency)>10):
            #print(sum/n,sum2/n)
            return([np.mean(inferencelatency),np.std(inferencelatency)])
        else:
            print("Duration time is too short!")


def interference():
    #batch=[3,9]
    #thread=[20,40]
    batch=[3,9,27]
    thread=[20,40,50,60,80]
    m=len(models)
    index=3000
    #1000 3000
    dur=[]
    std=[]
    for i in range(m):
        for j in range(i+1,m):
            b=len(batch)
            for b1 in batch:
                for b2 in batch:
                    for th1 in thread:
                        th2=100-th1
                        subprocess.run("./pairinference.sh "+str(index)+" "+models[i]+" "+str(th1)+" "+str(b1)+" "\
                            +models[j]+" "+str(th2)+" "+str(b2), shell=True)
                        dur1=durationps("data/durtime_"+str(index)+"_"+models[i])
                        dur2=durationps("data/durtime_"+str(index)+"_"+models[j])
                        dur.append([models[i],models[j],b1,b2,th1,dur1[0],dur2[0]])
                        std.append([models[i],models[j],b1,b2,th1,dur1[1],dur2[1]])
                        index+=1
    saverecords("interference","latency",dur)
    saverecords("interference","std",std)

if __name__ == '__main__':
    #for i in models:
    #    soloduration(i)
    interference()



