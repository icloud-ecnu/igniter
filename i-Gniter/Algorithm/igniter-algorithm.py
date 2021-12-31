import json
import copy
from operator import index
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import append
from scipy.optimize import curve_fit
from scipy import interpolate 
import math
import copy

class Context:
    frequency=1530
    power=300
    idlepower=52
    batchsize=[1,16,32]
    thread=[10,50,100]
    models=["alexnet","resnet50","vgg19","ssd"]
    kernels=[20,80,29,93]
    unit=2.5
    step=40
    bandwidth=15
    '''
    def __init__(self,f,po):
        self.frequency = f
        self.power = po
    '''
class Model:
    name="alexnet"
    kernels=29
    #power=100
    baseidle=0.1
    #act_popt[0-4] k1-k5
    act_popt=[]
    l2cache=0
    #powers=[]
    power_popt=[]
    l2cache_popt=[]
    k_l2=1
    tmpl2cache=0
    inputdata=1000
    outputdata=1000
    #batch size 1
    def p(self):
        print(self.name,self.kernels,self.baseidle,self.act_popt,self.l2cache,self.k_l2)


context=Context()

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

def p_f(po,a):
    #power to frequency when frequency>1530
    return a*(po-context.power)+context.frequency

def power_frequency(power,frequency):
    #the idle percent of VGG19 is very low, the GPU power of VGG-19 grows linearly. 
    #p_f(power,popt[0])
    k=0
    l=0
    for i in range(len(frequency)):
        if(frequency[i]==1530):
            if(i>=1):
                k+=(power[i]-power[i-1])
                l+=1
        else:
            k/=l
            for j in range(i,len(power)):
                power[j]=power[j-1]+k
            popt, pcov = curve_fit(p_f, power[i:], frequency[i:])
            #print(power,frequency)
            #yp = p_f(330,popt[0])
            #print(yp)
            return popt

def idletime(idletime,frequency):
    # use : np.polyval(f1, number of processes)
    # vgg19
    l=len(idletime)
    vgg19=[0]*l
    for i in range(l):
        vgg19[i]=idletime[i]*frequency[i]/1530
    #vgg19=[i/29 for i in idletime]
    fp=[]
    x=np.array([2,3,4,5])
    for i in range(1,len(vgg19)):
        fp.append(vgg19[i]-vgg19[0])
    fp=np.array(fp)
    #print(x,fp)
    f1 = np.polyfit(x, fp, 1)
    yvals=np.polyval(f1, [2,3,4,5])
    #print(f1)
    #print(yvals)
    return f1

def activetime_func(x,k1,k2,k3,k4,k5):
    r=(k1*x[0]**2+k2*x[0]+k3)/(x[1]+k4)+k5
    #return r.ravel()
    return r
'''
def activetime_func(x,k2,k3,k4,k5):
    r=(k2*x[0]+k3)/(x[1]+k4)+k5
    #return r.ravel()
    return r
'''
def compute_idletime(kernels):
    return 1


def activetime(gpulatency,frequency,baseidletime):
    #activetime_func([b,th], popt[0], popt[1], popt[2], popt[3], popt[4])
    activelatency=[]
    x=[[],[]]
    #batchsize thread
    for i in range(len(context.batchsize)):
        for j in range(len(context.thread)):
            tmp=gpulatency[i][j]
            if(frequency[i][j]<context.frequency):
                tmp*=(frequency[i][j]/context.frequency)
            tmp-=baseidletime
            activelatency.append(tmp)
            x[0].append(context.batchsize[i])
            x[1].append(context.thread[j])
    #print(x,activelatency)
    #print(np.array(x),np.array(activelatency))
    popt, pcov = curve_fit(activetime_func, np.array(x), np.array(activelatency), maxfev=100000)
    #yp = activetime_func([1,100], popt[0], popt[1], popt[2], popt[3], popt[4])
    #print(yp)
    return popt

def ability_cp(x,k,b):
    return k*x+b

def power_l2cache(power,gpulatency,baseidletime,idlepower,l2caches,frequency):
    #power gpulatecny t50 
    #power[batchsize][thread]
    #popt act_popt
    batch=[1,16,32]
    resource=[10,50,100]
    ability=[]
    ability_p=[]
    ypower=[]
    for i in range(3):
        for j in range(3):
            b=batch[i]
            r=resource[j]
            ability.append((1000*b)/(gpulatency[i][j]-baseidletime))
            if frequency[i][j]>1529:
                ability_p.append((1000*b)/(gpulatency[i][j]-baseidletime))
                ypower.append(power[i][j]-idlepower)
    #print(ability_p, ypower)
    #print([ability[0],ability[4],ability[8]], l2caches)
    popt_p, pcov_p = curve_fit(ability_cp, ability_p, ypower)
    popt_c, pcov_c = curve_fit(ability_cp, [ability[0],ability[4],ability[8]], l2caches)
    
    return popt_p, popt_c

def solo_power(popt,power,gpulatency,baseidletime,p_f):
    #power gpulatecny t50 
    #power[thread][batchsize]
    #popt act_popt
    power_t50=[]
    gpulatency_t50=[]
    for i in range(len(context.batchsize)):
        power_t50.append(power[i][1]-context.idlepower)
        gpulatency_t50.append(gpulatency[i][1])
    #power_t50=[i-context.idlepower for i in power]
    l=len(gpulatency_t50)
    th_t50=[context.batchsize[i]/gpulatency_t50[i]*1000 for i in range(l)]
    k=(power_t50[l-1]-power_t50[0])/(th_t50[l-1]-th_t50[0])
    solo_power={}
    solo_power[50.0]={}
    for i in range(1,context.batchsize[-1]+1):
        act=activetime_func([i,50], popt[0], popt[1], popt[2], popt[3], popt[4])
        th=i/(act+baseidletime)*1000
        solo_power[50.0][i]=(th-th_t50[0])*k+power_t50[0]
        #print(solo_power[50.0][i],throughput,throughput_t50[0],power_t50[0],k)
        for j in range(1,context.step+1):
            tmpt=j*context.unit
            if tmpt not in solo_power:
                solo_power[tmpt]={}
            if(tmpt!=50.0):
                #print(tmpt)
                solo_power[tmpt][i]=act*solo_power[50.0][i]/activetime_func([i,tmpt], popt[0], popt[1], popt[2], popt[3], popt[4])

    #print(solo_power)
    return solo_power

def predict(models,batchsize,thread,idlef,powerp):
    # batchsize [[gpulatency] ,[inference latency]]
    p=len(models)
    ans=[[],[]]
    increasing_idle=0
    if(p>=2):
        increasing_idle=np.polyval(idlef, p)
    total_l2cache=0
    total_power=context.idlepower
    frequency=context.frequency
    tmpcache=[]
    for i in range(len(models)):
        m=models[i]
        tact_solo=activetime_func([batchsize[i],thread[i]], m.act_popt[0], m.act_popt[1], m.act_popt[2], m.act_popt[3], m.act_popt[4])
        tmppower=ability_cp((1000*batchsize[i])/tact_solo,m.power_popt[0],m.power_popt[1])
        total_power+=tmppower
        tmpl2cache=ability_cp((1000*batchsize[i])/tact_solo,m.l2cache_popt[0],m.l2cache_popt[1])
        tmpcache.append(tmpl2cache)
        total_l2cache+=tmpl2cache
    if(total_power>300):
        frequency=p_f(total_power,powerp[0])
    for i in range(len(models)):
        m=models[i]
        idletime=m.kernels*increasing_idle+m.baseidle
        #print("idletime",idletime,m.kernels,m.baseidle,increasing_idle)
        tact_solo=activetime_func([batchsize[i],thread[i]], m.act_popt[0], m.act_popt[1], m.act_popt[2], m.act_popt[3], m.act_popt[4])
        l2=total_l2cache-tmpcache[i]
        activetime=tact_solo*(1+m.k_l2*l2)
        gpu_latency=(idletime+activetime)/(frequency/context.frequency)
        #print(idletime,frequency,context.frequency)
        #print("inference_latency",inference_latency)
        #print(gpu_latency)
        ans[0].append(gpu_latency)
        ans[1].append(gpu_latency+(m.inputdata+m.outputdata)*batchsize[i]/context.bandwidth)
        #print(total_power,tact_solo,activetime,idletime)
    #print("latency",ans)
    return ans


def error(act,pre):
    err=[]
    for i in range(len(act)):
        err.append(abs(act[i]-pre[i])/act[i])
        #print(act[i],pre[i],err[i])
    return err

def durationps(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        log=f.readlines()
        gpulatency=[]
        inferencelatency=[]
        for i in range(1, len(log)-1):
            j=eval(log[i][1:])
            #j = json.loads(log[i])
            if(j["startComputeMs"]>10000 and j["endComputeMs"]<20000):
                #latencyMs computeMs
                gpulatency.append(j["computeMs"])
                inferencelatency.append(j["latencyMs"])
        if(len(gpulatency)>10):
            #print(sum/n,sum2/n)
            return([np.mean(gpulatency),np.mean(inferencelatency)],[np.std(gpulatency),np.std(inferencelatency)])
        else:
            print("Duration time is too short!")


def canadded(inference,batch,resource,idlef,powerp,slo):
    addinferece=[]
    addbatch=[]
    addresource=[]
    for mi in range(len(resource)):
        if(resource[mi]>0):
            addinferece.append(inference[mi])
            addbatch.append(batch[mi])
            addresource.append(resource[mi])
    ans=[True] * len(inference)
    #if(len(addinferece)==1):
    #    return ans
    latency=predict(addinferece,addbatch,addresource,idlef,powerp)[1]
    #print("predict",latency,addbatch,addresource)
    li=0
    for mi in range(len(resource)):
        #print(resource,slo)
        #print(latency)
        if(resource[mi]>0):
            if(latency[li]>slo[mi]):
                ans[mi]=False
            li+=1
        #if(resource>100):
        #    print("workload is too large")
    #print("tr",ans)
    return ans

def algorithm(inference,slo,arrivate,idlef,powerp):
    #inference: models
    slo=[i/2 for i in slo]
    m=len(inference)
    resource=[[0] * m for i in range(m)]
    batch=[0] * m
    r_lower=[[i,0] for i in range(m)]
    #print("batch",m,batch)
    # arrivate ips 
    arrivate=[i/1000 for i in arrivate]
    for i in range(m):
        inf=inference[i]
        #print("slo",slo[i],"ar",arrivate[i],"b",context.bandwidth,"data",inf.inputdata,inf.outputdata)
        batch[i]=math.ceil(slo[i]*arrivate[i]*context.bandwidth/(context.bandwidth+inf.inputdata*arrivate[i]))
        gama=inf.act_popt[0]*batch[i]**2+inf.act_popt[1]*batch[i]+inf.act_popt[2]
        delta=slo[i]-inf.baseidle-batch[i]*(inf.inputdata+inf.outputdata)/context.bandwidth-inf.act_popt[4]
        #print(gama-inf.act_popt[4]*delta,inf.act_popt[3]*delta*context.unit)
        r_lower[i][1]=math.ceil((gama/delta-inf.act_popt[3])/context.unit)*context.unit
    v=1
    for r_l in r_lower:
        if(r_l[1]>100):
            print("workload is too large!")
            return
    #print("batch",batch)
    #print("r_lower",r_lower)
    r_lower.sort(key=lambda x:x[1], reverse=True)
    for ti in r_lower:
        i=ti[0]
        #print(ti,"ti")
        inter=[100] * v
        ra_ij=copy.deepcopy(resource)
        flag=False
        for j in range(0,v):
            ra_j=0
            for inf in range(m):
                ra_j+=resource[inf][j]
            r_t=ra_j+ti[1]
            ra_ij[i][j]=ti[1]
            #print("ra_j[j]",ra_j,"i",i,"j",j,"ra_ij",ra_ij,"resource",resource)
            #print("r_t",r_t)
            #ad=0
            while(r_t<101):
                #tmpinferece=copy(inference)
                #tmpbatch=copy(batch)
                tmpresource=[tmp[j] for tmp in ra_ij]
                add=canadded(inference,batch,tmpresource,idlef,powerp,slo)
                c=True
                for d in add:
                    c=c and d
                #print("c",c)
                #print("r_t",r_t,c)
                #if(ad>1):
                #    break
                #ad+=1
                if(c):
                    inter[j]=r_t-(ra_j+ti[1])
                    flag=True
                    #print("inter",inter[j])
                    break
                else:
                    for tmp in range(m):
                        if(not add[tmp]):
                            ra_ij[tmp][j]+=context.unit
                            r_t+=context.unit
        mini=min(inter)
        #print("mini",mini,resource,flag)
        j=inter.index(mini)
        #print("jv",inter,v)
        if(flag):
            for inf in range(m):
                resource[inf][j]=ra_ij[inf][j]
        else:
            v+=1
            #print(i,"r_lower",r_lower,v)
            resource[i][v-1]=ti[1]
        #print("resource",resource)
    #print("batch",batch)
    #print("resource",resource)
    lr=len(resource)
    #print("r_lower",r_lower)
    for i in range(lr):
        print(i,slo[i],arrivate[i]*1000)
    print("id,model,batch,resources")
    for i in range(lr):
        ans=[]
        for j in range(lr):
            if(resource[j][i]!=0):
                ans.append([j,inference[j].name,batch[j],resource[j][i]])
        if(ans==[]):
            break
        print("GPU",i+1)
        print(ans)
        

if __name__ == '__main__':
    profile=loadprofile("./config")

    context.bandwidth=10000000

    context.idlepower=profile["hardware"]["idlepower"]
    powerp=power_frequency(profile["hardware"]["power"], profile["hardware"]["frequency"])
    idlef=idletime(profile["hardware"]["idletime"],profile["hardware"]["frequency"])
    #print(context.idlepower,context.bandwidth,idlef,powerp)
    model_par={}
    models=[]
    model2={}
    j=0
    for i in context.models:
        m=Model()
        m.name=i
        m.kernels=context.kernels[j]
        m.baseidle=profile[i]["idletime_1"]
        m.inputdata=profile[i]["inputdata"]
        m.outputdata=profile[i]["outputdata"]
        j+=1
        m.l2cache=profile[i]["l2cache"]
        #m.k_l2=((profile[i]["activetime_5"]*profile[i]["frequency_5"]/context.frequency)/profile[i]["activetime_1"]-1)/(4*m.l2cache)
        #print(profile[i]["activetime_2"],profile[i]["frequency_2"],context.frequency,profile[i]["activetime_1"],m.l2cache)
        m.k_l2=((profile[i]["activetime_2"]*profile[i]["frequency_2"]/context.frequency)/profile[i]["activetime_1"]-1)/(m.l2cache)
        #print("L2",m.k_l2)
        m.act_popt=activetime(profile[i]["gpulatency"],profile[i]["frequency"],profile[i]["idletime_1"])
        #m.act_popt=profile[i]["popt"]
        #m.powers=solo_power(m.act_popt,profile[i]["power"],profile[i]["gpulatency"],profile[i]["idletime_1"],powerp)
        m.power_popt, m.l2cache_popt=power_l2cache(profile[i]["power"],profile[i]["gpulatency"],m.baseidle,context.idlepower,profile[i]["l2caches"],profile[i]["frequency"])
        models.append(m)
        model2[i]=m
        #m.p()

    #motivation example
    #model0: AlexNet, model1: ResNet-50, model2: VGG-19
    models=[models[0],models[1],models[2]]
    SLOs=[15,40,60]
    rates=[500,400,200]

    algorithm(models,SLOs,rates,idlef,powerp)
