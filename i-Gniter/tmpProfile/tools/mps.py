import subprocess

from numpy.lib.function_base import copy
import gslice
import logging
import numpy as np
import json

def testans(inference,batch):
    l=len(inference)
    index=100001
    shell="./testinference.sh "+str(index)+" "
    for i in range(l):
        # resource 25
        shell+=(inference[i]+" 100 "+str(batch[i])+" ")
    subprocess.run(shell, shell=True)
    observe=[]
    for i in range(l):
        o=gslice.durationps("data/durtime_"+str(index)+"_"+inference[i])[0]
        o[0]*=batch[i]
        observe.append(o)
    print(observe)

def findbatch(inference,index,slo):
    l=len(inference)
    batch=[1]*l
    inferes=[]
    for i in range(l):
        i1=gslice.Infere()
        i1.batch.append(1)
        i1.resource.append(100)
        inferes.append(i1)

    for t in range(10):
        shell="./testinference.sh "+str(index)+" "
        for i in range(l):
            # resource 25
            shell+=(inference[i]+" 100 "+str(batch[i])+" ")
        #logging.info("%s",shell)
        subprocess.run(shell, shell=True)
        flag=True
        for i in range(l):
            observe=gslice.durationps("data/durtime_"+str(index)+"_"+inference[i])[0]
            inferes[i].latency.append(observe[1])
            inferes[i].throughput.append(observe[0])
            print(batch[i])
            curbatch=batch[i]
            batch[i]=gslice.slab(slo[i],observe[1],batch[i])
            #logging.info("avg_latency %s",str(observe[1]))
            #logging.info("SLAB %s %s",str(batch[i]),str(resource[i]))
            inferes[i].batch.append(batch[i])
            inferes[i].resource.append(100)
            if(batch[i]!=curbatch):
                flag=False
        index+=1
        if(flag):
            break
    return inferes

def saverecords(kind,metrics,value):
    path="mpsc"
    records={kind:{metrics:value}}
    json_str=json.dumps(records)
    #print(json_str)
    with open(path,"a") as file:
        file.write(json_str+"\n")

def subsets(sets):
    l=len(sets)
    subsets=[]
    s=1<<l
    for i in range(1,s):
        j=0
        tmp=[]
        while(i>0):
            if(i%2==1):
                tmp.append(sets[j])
            i=i>>1
            j+=1
        subsets.append(tmp)
    print(subsets)
    return subsets

def place(model,li,placeplan):
    newli=[]
    tmppp=copy(placeplan)
    for i in li:
        tmppp.append(model[i])
        flag=True
        for ni in li:
            for j in model[i]:
                for k in model[ni]:
                    if(model[ni][k]!=model[i][j]):
                        flag=False


def strategy(model,throughput,arrivalrate):
    arrivalrate=1
    l=len(model)
    li=[]
    for i in range(l):
        flag=True
        for j in len(model[i]):
            if(throughput[i][j]<arrivalrate[i][j]):
                flag=False
                break
        if(flag):
            li.append(i)
    vm=0
    plan=[]
    for i in model[-1]:
        for j in li:
            for k in model[j]:
                if(i==k):
                    k=1
    

    



if __name__ == '__main__':
    models=["alexnet","resnet50","vgg19","bert"]
    slos=[5,20,30,60]
    #testans([models[0],models[1]],[17.5,35],[5,10])
    #testans([models[0],models[1],models[2]],[15,32.5,37.5],[10,8,8])
    #testans([models[2],models[3]],[52.5,47.5],[18,12])
    model=subsets(models)
    slo=subsets(slos)
    anses=[]
    #index=3000
    index=2800
    for i in range(0,6):
        ans=findbatch(model[i],index,slo[i])
        index+=20
        anses.append(ans)
    k=0
    for ans in anses:
        print(model[k])
        k+=1
        for j in ans:
            gslice.Infere.printth(j)
        
