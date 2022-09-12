import json
import numpy as np
from scipy.optimize import curve_fit
import math
import copy

class Context:
    frequency=1530
    power=300
    idlepower=52
    batchsize=[1,16,32]
    thread=[10,50,100]
    models=["alexnet_dynamic","resnet50_dynamic","vgg19_dynamic","ssd_dynamic"]
    kernels=[20,80,29,93]
    unit=2.5
    step=40
    bandwidth=10
    powerp=[]
    idlef=[]

class Model:
    name="alexnet"
    kernels=29
    baseidle=0.1
    #act_popt[0-4] k1-k5
    act_popt=[]
    l2cache=0
    power_popt=[]
    l2cache_popt=[]
    k_l2=1
    tmpl2cache=0
    inputdata=1000
    outputdata=1000


context=Context()

def loadprofile(path):
    profile={}
    with open(path,"r") as file:
        for line in file:
            data=json.loads(line)
            for key in data:
                if(key not in profile):
                    profile[key]=data[key]
                else:
                    profile[key].update(data[key])

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
            return popt

def idle_f(n_i,alpha_sch,beta_sch):
    return alpha_sch*n_i+beta_sch

def idletime(idletime,frequency):
    # use : idle_f(number of inference workloads, popt[0], popt[1])
    # vgg19
    l=len(idletime)
    vgg19=[0]*l
    for i in range(l):
        vgg19[i]=idletime[i]*frequency[i]/1530
    fp=[]
    x=np.array([2,3,4,5])
    for i in range(1,len(vgg19)):
        fp.append(vgg19[i]-vgg19[0])
    fp=np.array(fp)
    popt, pcov = curve_fit(idle_f, x, fp)
    return popt

def activetime_func(x,k1,k2,k3,k4,k5):
    r=(k1*x[0]**2+k2*x[0]+k3)/(x[1]+k4)+k5
    return r



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
    popt, pcov = curve_fit(activetime_func, np.array(x), np.array(activelatency), maxfev=100000)
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
    popt_p, pcov_p = curve_fit(ability_cp, ability_p, ypower)
    popt_c, pcov_c = curve_fit(ability_cp, [ability[0],ability[4],ability[8]], l2caches)
    
    return popt_p, popt_c

def predict(models,batchsize,thread):
    # batchsize [[throughput] ,[inference latency]]
    p=len(models)
    ans=[[],[]]
    increasing_idle=0
    if(p>=2):
        increasing_idle=idle_f(p, context.idlef[0], context.idlef[1])
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
        frequency=p_f(total_power,context.powerp[0])
    for i in range(len(models)):
        m=models[i]
        idletime=m.kernels*increasing_idle+m.baseidle
        tact_solo=activetime_func([batchsize[i],thread[i]], m.act_popt[0], m.act_popt[1], m.act_popt[2], m.act_popt[3], m.act_popt[4])
        l2=total_l2cache-tmpcache[i]
        activetime=tact_solo*(1+m.k_l2*l2)
        gpu_latency=(idletime+activetime)/(frequency/context.frequency)

        ans[0].append(batchsize[i]*1000/(gpu_latency+m.outputdata*batchsize[i]/context.bandwidth))
        ans[1].append(gpu_latency+(m.inputdata+m.outputdata)*batchsize[i]/context.bandwidth)

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


def r_total(ra_j):
    ans=0
    for r in ra_j:
        ans+=r[1]
    return ans


def alloc_gpus(inference,batch,slo,ra_ij,r_lower,w,j):
    flag=1
    ra_ij[j].append([w,r_lower])
    if(r_total(ra_ij[j])>100):
        return ra_ij
    addinferece=[]
    addbatch=[]
    addresource=[]

    num_workloads=len(ra_ij[j])
    for infwork in ra_ij[j]:
        
        addinferece.append(inference[infwork[0]])
        addbatch.append(batch[infwork[0]])
        addresource.append(infwork[1])

    while(np.sum(addresource)<=100 and flag==1):
        flag=0
        latency=predict(addinferece,addbatch,addresource)[1]
        for i in range(num_workloads):
            if(latency[i]>slo[ra_ij[j][i][0]]):
                addresource[i]+=context.unit
                flag=1
    
    for i in range(num_workloads):
        ra_ij[j][i][1]=addresource[i]
    return ra_ij


def algorithm(inference,slo,rate):
    #inference: models
    slo=[i/2 for i in slo]
    m=len(inference)
    resource=[[]]
    batch=[0] * m
    r_lower=[[i,0] for i in range(m)]
    # arrivate ips 
    arrivate=[i/1000 for i in rate]
    for i in range(m):
        inf=inference[i]
        batch[i]=math.ceil(slo[i]*arrivate[i]*context.bandwidth/(context.bandwidth+inf.inputdata*arrivate[i]))
        gama=inf.act_popt[0]*batch[i]**2+inf.act_popt[1]*batch[i]+inf.act_popt[2]
        delta=slo[i]-inf.baseidle-batch[i]*(inf.inputdata+inf.outputdata)/context.bandwidth-inf.act_popt[4]
        r_lower[i][1]=math.ceil((gama/delta-inf.act_popt[3])/context.unit)*context.unit
    g=1
    for r_l in r_lower:
        if(r_l[1]>100):
            print("workload is too large!")
            return
    r_lower.sort(key=lambda x:x[1], reverse=True)
    for ti in r_lower:
        i=ti[0]
        r_l=ti[1]
        inter=[100] * g
        ra_ij=copy.deepcopy(resource)
        q=-1
        r_inter_min=101
        for j in range(g):
            ra_ij=alloc_gpus(inference,batch,slo,ra_ij,r_l,i,j)
            r_inter=r_total(ra_ij[j])-r_total(resource[j])
            if(r_total(ra_ij[j])<=100 and r_inter<r_inter_min):
                q=j
                r_inter_min=r_inter
        if(q==-1):
            g+=1
            resource.append([])
            resource[g-1].append([i,r_l])
        else:
            resource[q]=copy.deepcopy(ra_ij[q])

        
    lr=len(resource)
    #print("r_lower",r_lower)
    for i in range(m):
        print(i,slo[i],rate[i])
    #print(resource)
    print("id,model,batch,resources")
    gpuid=1
    for gpu in resource:
        ans=[]
        conf={"models":[],"rates":[],"slos":[],"resources":[],"batches":[]}
        for inf in gpu:
            ans.append([inf[0],inference[inf[0]].name,batch[inf[0]],inf[1]])
            conf["models"].append(inference[inf[0]].name)
            conf["rates"].append(rate[inf[0]])
            conf["slos"].append(slo[inf[0]])
            conf["resources"].append(inf[1])
            conf["batches"].append(inf[0])
        if(ans==[]):
            break

        print("GPU",gpuid)
        print(ans)
        with open("./config_gpu"+str(gpuid)+".json","w") as f:
            json.dump(conf,f)
        gpuid+=1
        
    

if __name__ == '__main__':
    profile=loadprofile("./config")

    context.bandwidth=10000000

    context.idlepower=profile["hardware"]["idlepower"]
    context.powerp=power_frequency(profile["hardware"]["power"], profile["hardware"]["frequency"])
    context.idlef=idletime(profile["hardware"]["idletime"],profile["hardware"]["frequency"])
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
        m.k_l2=((profile[i]["activetime_2"]*profile[i]["frequency_2"]/context.frequency)/profile[i]["activetime_1"]-1)/(m.l2cache)
        m.act_popt=activetime(profile[i]["gpulatency"],profile[i]["frequency"],profile[i]["idletime_1"])
        m.power_popt, m.l2cache_popt=power_l2cache(profile[i]["power"],profile[i]["gpulatency"],m.baseidle,context.idlepower,profile[i]["l2caches"],profile[i]["frequency"])
        models.append(m)
        model2[i]=m


    #motivation example
    #model0: AlexNet, model1: ResNet-50, model2: VGG-19
    workloads=[models[0],models[1],models[2]]
    SLOs=[15,40,60]
    rates=[500,400,200]
    algorithm(workloads,SLOs,rates)
