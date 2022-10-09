import json
import copy
from operator import index
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from numpy.core.fromnumeric import sort
from numpy.lib.function_base import append
from scipy.optimize import curve_fit
from scipy import interpolate 
import sys
import subprocess
import datetime
import trtexecps
import math
import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import numpy.linalg as LA

def loadprofile(path):
    profile={}
    with open(path,"r") as file:
        for line in file:
            data=json.loads(line)
            for key in data:
                #profile.update(data)
                if(key not in profile):
                    profile[key]=data[key]
                    #profile[key]=copy.deepcopy(data[key])
                else:
                    #print(profile[key],data[key])
                    profile[key].update(data[key])
                #print(profile)
    #print(profile)
    return profile

def error(act,pre):
    return abs(act-pre)/act


def testinterference(rindex,rs,lrModel,profile):
    j=0
    latency=test["interference"]["latency"]
    errors=[]
    for i in range(len(rindex)):
        if(rindex[i]==rs[j]):
            j+=1
        else:
            lat=latency[i]
            m1=lat[0]
            m2=lat[1]
            b1=str(lat[2])
            th1=str(lat[4])
            b2=str(lat[3])
            th2=str(100-lat[4])
            inter1=lrModel.predict([[profile[m1]["l2cache"][b1][th1],profile[m2]["l2cache"][b2][th2],profile[m1]["dram"][b1][th1],profile[m2]["dram"][b2][th2]]])
            inter2=lrModel.predict([[profile[m2]["l2cache"][b2][th2],profile[m1]["l2cache"][b1][th1],profile[m2]["dram"][b2][th2],profile[m1]["dram"][b1][th1]]])
            print(profile[m1]["latency"][b1][th1],inter1[0][0],lat[5],profile[m1]["latency"][b1][th1]*(1+inter1[0][0]))
            print(profile[m2]["latency"][b2][th2],inter2[0][0],lat[6],profile[m2]["latency"][b2][th2]*(1+inter2[0][0]))
            errors.append(error(lat[5],profile[m1]["latency"][b1][th1]*(1+inter1[0][0])))
            errors.append(error(lat[6],profile[m2]["latency"][b2][th2]*(1+inter2[0][0])))
    print(np.mean(errors),np.std(errors))


def testinterference2(lrModel,profile,test):
    batch1=[3,9,27]
    batch2=[10,20,30]
    models=["alexnet","resnet50","vgg19","ssd"]
    resource=[20,40,50,60,80]
    latency=test["interference"]["latency"]
    errors=[]
    errori={}
    i=0
    for i in range(len(latency)):
        lat=latency[i]
        m1=lat[0]
        m2=lat[1]
        b1=str(lat[2])
        th1=str(lat[4])
        b2=str(lat[3])
        th2=str(100-lat[4])
        inter1=lrModel.predict([[profile[m1]["l2cache"][b1][th1],profile[m2]["l2cache"][b2][th2],profile[m1]["dram"][b1][th1],profile[m2]["dram"][b2][th2]]])
        inter2=lrModel.predict([[profile[m2]["l2cache"][b2][th2],profile[m1]["l2cache"][b1][th1],profile[m2]["dram"][b2][th2],profile[m1]["dram"][b1][th1]]])
        pre1=profile[m1]["latency"][b1][th1]*(1+inter1[0][0])
        pre2=profile[m2]["latency"][b2][th2]*(1+inter2[0][0])
        #print(profile[m1]["latency"][b1][th1],inter1[0][0],lat[5],pre1)
        #print(profile[m2]["latency"][b2][th2],inter2[0][0],lat[6],pre2)
        errors.append(error(lat[5],pre1))
        errors.append(error(lat[6],pre2))

        errori[lat[0]]=abs(pre1-lat[5])/lat[5]
        errori[lat[1]]=abs(pre2-lat[6])/lat[6]
    print(errori)
    print(np.mean(errors),np.std(errors))

def initinterference(profile,test):
    ans=[]
    if(False):
        latency=[]
        for la in profile["interference"]["latency"]:
            if(la[0]!="ssd" and la[1]!="ssd"):
                latency.append(la)
        l=len(latency)
        print(l)
    l=len(profile["interference"]["latency"])
    latency=profile["interference"]["latency"]
    print(l)

    rl=int(0.7*l)
    rindex=[i for i in range(l)]
    rs=random.sample(rindex,rl)
    x=[]
    y=[]
    for i in rs:
        lat=latency[i]
        m1=lat[0]
        m2=lat[1]
        b1=str(lat[2])
        th1=str(lat[4])
        b2=str(lat[3])
        th2=str(100-lat[4])
        x.append([profile[m1]["l2cache"][b1][th1],profile[m2]["l2cache"][b2][th2],profile[m1]["dram"][b1][th1],profile[m2]["dram"][b2][th2]])
        x.append([profile[m2]["l2cache"][b2][th2],profile[m1]["l2cache"][b1][th1],profile[m2]["dram"][b2][th2],profile[m1]["dram"][b1][th1]])
        #model name
        y.append([lat[5]/profile[m1]["latency"][b1][th1]-1])
        y.append([lat[6]/profile[m2]["latency"][b2][th2]-1])
    lrModel = linear_model.LinearRegression()
    #print(x,y)
    lrModel.fit(x,y)
    testinterference2(lrModel,profile,test)
    return lrModel

def findbestfit(remain_gpulets,p_ideal,slo,lrModel,model,profile,alloc_gpulets,request):
    remain_gpulets.sort(key=lambda x:x[1])
    resource=0
    batch=0
    for i in range(len(remain_gpulets)):
        gpulet=remain_gpulets[i][1]
        if gpulet >= p_ideal:
            v=remain_gpulets[i][0]
            resource=gpulet
            inter=0
            if gpulet==100:
                resource=p_ideal
                remain_gpulets[i][1]-=p_ideal
                print(model,str(batch+1),str(resource),slo)
                while(batch<32 and profile[model]["latency"][str(batch+1)][str(resource)]<slo):
                    batch+=1
            else:
                ag=alloc_gpulets[v][0]
                b=str(ag[1])
                r=str(ag[2])
                inter=lrModel.predict([[profile[ag[0]]["l2cache"][b][r],profile[model]["l2cache"][str(batch+1)][str(resource)],\
                    profile[ag[0]]["dram"][b][r],profile[model]["dram"][str(batch+1)][str(resource)]]])
                while(batch<32 and profile[model]["latency"][str(batch+1)][str(gpulet)]*(1+inter)<slo):
                    batch+=1
                    inter=lrModel.predict([[profile[ag[0]]["l2cache"][b][r],profile[model]["l2cache"][str(batch+1)][str(resource)],\
                    profile[ag[0]]["dram"][b][r],profile[model]["dram"][str(batch+1)][str(resource)]]])
                remain_gpulets.pop(i)
            alloc_gpulets[v].append([model,batch,resource,request])
            break
    return [batch,resource]


def gpulet(inference,slo,arrivate,vm,MAXEFFICIENT,lrModel,profile):
    m=len(inference)
    arrivate_n=[[i,arrivate[i]] for i in range(m)]
    arrivate_n.sort(key=lambda x:x[1], reverse=True)
    alloc_gpulets=[[] for i in range(vm) ] 
    #0:[[i,batch,resource],[i,batch,resource]],1:
    remain_gpulets=[[i,100] for i in range(vm)]
    for ns in arrivate_n:
        i=ns[0]
        incoming_rate=ns[1]
        assigned_rate=0
        pro=[]
        while (assigned_rate<incoming_rate):
            p_eff=MAXEFFICIENT[inference[i]]
            p_req=100
            rate_m=incoming_rate-assigned_rate
            resource=[10,20,40,50,60,80,100]
            for r in resource:
                if(profile[inference[i]]["throughput"]["1"][str(r)]>rate_m):
                    p_req=r
            p_ideal=min(p_eff,p_req)
            print(remain_gpulets,p_ideal,slo[i],lrModel,inference[i],alloc_gpulets)
            br=findbestfit(remain_gpulets,p_ideal,slo[i],lrModel,inference[i],profile,alloc_gpulets,i)
            print(br)
            if(br==[0,0]): 
                return []
            assigned_rate+=profile[inference[i]]["throughput"][str(br[0])][str(br[1])]
            if(len(remain_gpulets)==0):
                return []
    return alloc_gpulets

def PJcurvature(x,y):
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    """
    t_a = LA.norm([x[1]-x[0],y[1]-y[0]])
    t_b = LA.norm([x[2]-x[1],y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0,    0     ],
        [1,  t_b, t_b**2]
    ])

    a = np.matmul(LA.inv(M),x)
    b = np.matmul(LA.inv(M),y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return abs(kappa), [b[1],-a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

def maxefficient(profile):
    #[10,20,30,40,50,60,70,80,90,100 ]
    resource=[10,20,40,50,60,80,100]
    models=["alexnet","resnet50","vgg19","ssd"]
    ka = []
    no = []
    po = []
    ans={}
    for m in models:
        rates=[]
        for r in resource:
            rates.append(profile[m]["throughput"]["1"][str(r)])
        ansre=0
        maxka=0
        for i in list(range(len(resource)-1))[1:]:
            #plt.plot([x2[i],y2[i]],'g*')
            x = [resource[i-1],resource[i],resource[i+1]]
            y = (rates[i-1],rates[i],rates[i+1])
            kappa,norm = PJcurvature(x,y)
            ka.append(kappa)
            no.append(norm)
            po.append([x[1],y[1]])
            #print(kappa)
            if(kappa>maxka):
                maxka=kappa
                ansre=resource[i]
                ans[m]=ansre
    return ans



if __name__ == '__main__':
    profile=loadprofile("./gpulet.config")
    test=loadprofile("./testcon.log")
    lrModel=initinterference(profile,test)
    if(False):
        models=["alexnet","resnet50","vgg19","ssd"]
        MAXEFFICIENT=maxefficient(profile)
        #[models[0],models[0],models[0],models[1],models[1],models[1],\
        #    models[2],models[2],models[2],models[3],models[3],models[3]],\
        #        [5,10,15,10,20,30,15,30,45,20,40,60],[1000,500,2000,400,600,200,300,400,200,200,100,300]
        inference=[models[0],models[0],models[0],models[1],models[1],models[1],\
            models[2],models[2],models[2],models[3],models[3],models[3]]
        slo=[5,7.5,10,5,10,15,10,15,20,15,20,25]
        arrivate=[1000,500,2000,400,600,200,300,400,200,200,100,300]
        plan=[]
        for i in range(1,20):
            g=gpulet(inference,slo,arrivate,i,MAXEFFICIENT,lrModel,profile)
            print(g)
            if(len(g)!=0):
                plan=g
                for vm in plan:
                    print(vm)
                print(i)
                break
       