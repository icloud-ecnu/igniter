import json
import copy
from operator import index
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from numpy.lib.function_base import append
from scipy.optimize import curve_fit
from scipy import interpolate 
import sys
import subprocess
import datetime
import trtexecps
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
    powers=[]
    k_l2=1
    tmpl2cache=0
    transferdata=1000
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
        vgg19[i]=idletime[i]*frequency[i]/1530/29
    #vgg19=[i/29 for i in idletime]
    fp=[]
    x=np.array([2,3,4,5])
    for i in range(1,len(vgg19)):
        fp.append(vgg19[i]-vgg19[0])
    fp=np.array(fp)
    #print(x,fp)
    f1 = np.polyfit(x, fp, 1)
    #yvals=np.polyval(f1, 2)
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
    print(np.array(x),np.array(activelatency))
    popt, pcov = curve_fit(activetime_func, np.array(x), np.array(activelatency), maxfev=100000)
    #yp = activetime_func([1,100], popt[0], popt[1], popt[2], popt[3], popt[4])
    #print(yp)
    return popt

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
    throughput_t50=[context.batchsize[i]/gpulatency_t50[i]*1000 for i in range(l)]
    k=(power_t50[l-1]-power_t50[0])/(throughput_t50[l-1]-throughput_t50[0])
    solo_power={}
    solo_power[50.0]={}
    for i in range(1,context.batchsize[-1]+1):
        act=activetime_func([i,50], popt[0], popt[1], popt[2], popt[3], popt[4])
        throughput=i/(act+baseidletime)*1000
        solo_power[50.0][i]=(throughput-throughput_t50[0])*k+power_t50[0]
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
    total_power=0
    frequency=context.frequency
    tmpcache=[]
    for i in range(len(models)):
        m=models[i]
        p=m.powers[thread[i]][batchsize[i]]
        total_power+=p
        m.tmpl2cache=p/m.powers[20][1]*m.l2cache
        tmpcache.append(m.tmpl2cache)
        total_l2cache+=m.tmpl2cache
    if(total_power>300):
        frequency=p_f(total_power,powerp[0])
    for i in range(len(models)):
        m=models[i]
        idletime=m.kernels*increasing_idle+m.baseidle
        #print("idletime",idletime,m.kernels,m.baseidle,increasing_idle)
        tact_solo=activetime_func([batchsize[i],thread[i]], m.act_popt[0], m.act_popt[1], m.act_popt[2], m.act_popt[3], m.act_popt[4])
        l2=total_l2cache-tmpcache[i]
        activetime=tact_solo*(1+m.k_l2*l2)
        #print("p",m.powers)
        gpu_latency=(idletime+activetime)/(frequency/context.frequency)
        #print(idletime,frequency,context.frequency)
        #print("inference_latency",inference_latency)
        #print(gpu_latency)
        ans[0].append(gpu_latency)
        ans[1].append(gpu_latency+m.transferdata*batchsize[i]/context.bandwidth)
        #print(total_power,tact_solo,activetime,idletime)
    print("latency",ans)
    return ans

def predict2(models,batchsize,thread,idlef,powerp,test,profile):
    # batchsize [[gpulatency] ,[inference latency]]
    p=len(models)
    ans=[[],[]]
    increasing_idle=0
    if(p>=2):
        increasing_idle=np.polyval(idlef, p)
    total_l2cache=0
    total_power=0
    frequency=context.frequency
    for i in range(len(models)):
        m_name=models[i].name
        b=str(batchsize[i])
        t=str(thread[i])
        p=test[m_name]["power"][b][t]
        total_power+=p
        total_l2cache+=profile[m_name]["l2cache"][b][t]
    if(total_power>300):
        frequency=p_f(total_power,powerp[0])
    for i in range(len(models)):
        m=models[i]
        m_name=models[i].name
        idletime=m.kernels*increasing_idle+m.baseidle
        #print("idletime",idletime,m.kernels,m.baseidle,increasing_idle)
        b=str(batchsize[i])
        t=str(thread[i])
        tact_solo=test[m_name]["gpulatency"][b][t]*(test[m_name]["frequency"][b][t]/context.frequency)-m.baseidle
        l2=total_l2cache-profile[m_name]["l2cache"][b][t]
        activetime=tact_solo*(1+m.k_l2*l2)
        #print("p",m.powers)
        gpu_latency=(idletime+activetime)/(frequency/context.frequency)
        #print(idletime,frequency,context.frequency)
        #print("inference_latency",inference_latency)
        #print(gpu_latency)
        ans[0].append(gpu_latency)
        ans[1].append(gpu_latency+m.transferdata*batchsize[i]/context.bandwidth)
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

def testpre(idlef,powerp,models):
    l=len(context.models)
    index=12000
    #gpu [[[model1, model2],[20],[30],...],[ij2],[ij3],...]
    act=[]
    act_std=[]
    pre=[]
    err=[]
    #inference
    act_inf=[]
    act_std_inf=[]
    pre_inf=[]
    err_inf=[]
    for i in range(l):
        for j in range(i+1,l):
            act.append([])
            act_std.append([])
            pre.append([])
            err.append([])
            act_inf.append([])
            pre_inf.append([])
            err_inf.append([])
            act_std_inf.append([])
            for th1 in range(10,100,10):
                th2=100-th1
                index+=1
                model1=context.models[i]
                model2=context.models[j]
                #subprocess.run("./testinference.sh "+str(index)+" "+model1+" "+str(th1)+" 1 "+model2+" "+str(th2)+" 1", shell=True)
                act[-1].append([durationps("data/durtime_"+str(index)+"_"+model1)[0][0],durationps("data/durtime_"+str(index)+"_"+model2)[0][0]])
                act_std[-1].append([durationps("data/durtime_"+str(index)+"_"+model1)[1][0],durationps("data/durtime_"+str(index)+"_"+model2)[1][0]])
                act_inf[-1].append([durationps("data/durtime_"+str(index)+"_"+model1)[0][1],durationps("data/durtime_"+str(index)+"_"+model2)[0][1]])
                act_std_inf[-1].append([durationps("data/durtime_"+str(index)+"_"+model1)[1][1],durationps("data/durtime_"+str(index)+"_"+model2)[1][1]])
                #print("data/durtime_"+str(index)+"_"+model1,model2)
                p=predict([models[i],models[j]],[1,1],[th1,th2],idlef,powerp)
                #print("p",p)
                pre[-1].append(p[0])
                pre_inf[-1].append(p[1])
                #print(act[-1][-1],pre[-1][-1])
                err[-1].append(error(act[-1][-1],pre[-1][-1]))
                err_inf[-1].append(error(act_inf[-1][-1],pre_inf[-1][-1]))
                #print(act,pre,err)
            #print(act)
            #print(pre)
            #print(err)
    #print(act)
    #print(pre)
    #print(err)
    #print(act_inf)
    print(act_std_inf)
    print(act_inf)
    print(pre_inf)
    print(err_inf)
    print(np.mean(err_inf))
    for i in range(len(act_inf)):
        for j in range(len(act_inf[0][0])):
            for k in range(len(act_inf[0])):
                print(act_inf[i][k][j],end=",")
            print()
            for k in range(len(act_inf[0])):
                print(pre_inf[i][k][j],end=",")
            print()
            for k in range(len(act_inf[0])):
                print(err_inf[i][k][j],end=",")
            print() 
            for k in range(len(act_inf[0])):
                print(act_std_inf[i][k][j],end=",")
            print()
        print()    

def nointerference():
    gpu_duration_t={}
    inf_duration_t={}
    for i in context.models:
        gpu_duration_t[i]=[]
        inf_duration_t[i]=[]
        for th1 in range(10,100,10):
            shell="./soloduration "+i+" "+str(th1)+" "+str(1)
            subprocess.run(shell, shell=True)
            outputpath="data/durtime_"+i+"_b"+str(1)+"_t"+str(th1)
            dur=trtexecps.trtexecps(outputpath)
            gpu_duration_t[i].append(dur[0])
            inf_duration_t[i].append(dur[1])
    gpu_duration_b={}
    inf_duration_b={}
    for i in context.models:
        gpu_duration_b[i]=[]
        inf_duration_b[i]=[]
        for b in [1,2,4,8,16,32]:
            shell="./soloduration "+i+" "+str(25)+" "+str(b)
            subprocess.run(shell, shell=True)
            outputpath="data/durtime_"+i+"_b"+str(b)+"_t"+str(25)
            dur=trtexecps.trtexecps(outputpath)
            gpu_duration_b[i].append(dur[0])
            inf_duration_b[i].append(dur[1])
    print(gpu_duration_t)
    print(inf_duration_t)
    print(gpu_duration_b)
    print(inf_duration_b)



def testpre2(idlef,powerp,models):
    l=len(context.models)
    index=13000
    act=[]
    act_std=[]
    pre=[]
    err=[]
    act_inf=[]
    act_std_inf=[]
    pre_inf=[]
    err_inf=[]
    #gpu [[model1,model2,3,4],[b2],[b3]]
    batchs=[1,2,4,8,16,32]
    for b in batchs:
        shell="./testinference.sh "+str(index)+" "
        act.append([])
        act_std.append([])
        #pre.append([])
        act_inf.append([])
        act_std_inf.append([])
        #pre_inf.append([])
        for m in context.models:
            # resource 25
            shell+=(m+" 25 "+str(b)+" ")
        #print(shell)
        #subprocess.run(shell, shell=True)
        for i in range(l):
            tmpdata=durationps("data/durtime_"+str(index)+"_"+context.models[i])
            act[-1].append(tmpdata[0][0])
            act_inf[-1].append(tmpdata[0][1])
            act_std_inf[-1].append(tmpdata[1][1])
        #print(models,[b,b,b,b],[25,25,25,25],idlef,powerp)
        p=predict(models,[b,b,b,b],[25,25,25,25],idlef,powerp)
        pre.append(p[0])
        pre_inf.append(p[1])
        #err=error(act[-1],pre[-1])
        err.append(error(act[-1],pre[-1]))
        err_inf.append(error(act_inf[-1],pre_inf[-1]))
        index+=1
    print(act_std_inf)
    print(act_inf)
    print(pre_inf)
    print(err_inf)
    print(np.mean(err_inf))
    for i in range(len(act_inf[0])):
        for j in range(len(act_inf)):
            print(act_inf[j][i],end=",")
        print()
        for j in range(len(act_inf)):
            print(pre_inf[j][i],end=",")
        print()
        for j in range(len(act_inf)):
            print(err_inf[j][i],end=",")
        print()
        for j in range(len(act_std_inf)):
            print(act_std_inf[j][i],end=",")
        print()        


def gpuletco(idlef,powerp,model2,contest):
    latency=contest["interference"]["latency"]
    errors=[]
    i=0
    errori={}
    for i in range(len(latency)):
        lat=latency[i]
        m1=model2[lat[0]]
        m2=model2[lat[1]]
        
        b1=lat[2]
        th1=lat[4]
        b2=lat[3]
        th2=100-lat[4]
        pre=predict([m1,m2],[b1,b2],[th1,th2],idlef,powerp)[1]
        #pre=predict2([m1,m2],[b1,b2],[th1,th2],idlef,powerp,solotest,profile)[1]
        #print(profile[m1]["latency"][b1][th1],inter1[0][0],lat[5],profile[m1]["latency"][b1][th1]*(1+inter1[0][0]))
        #print(profile[m2]["latency"][b2][th2],inter2[0][0],lat[6],profile[m2]["latency"][b2][th2]*(1+inter2[0][0]))
        print([lat[5],lat[6]],pre)
        errors.append(error([lat[5],lat[6]],pre))
        errori[lat[0]]=abs(pre[0]-lat[5])/lat[5]
        errori[lat[1]]=abs(pre[1]-lat[6])/lat[6]
    print(errori)
    print(np.mean(errors),np.std(errors))


def test(idlef,powerp,models):
    p=predict(models,[1,2,3,4],[25,32.5,15,17.5],idlef,powerp)
    return p

def analy(idlef,powerp,models):
    for b in [1,2,4,8,16,32]:
        for th1 in [10,25,50,100]:
            th2=100-th1
            #p=predict([models[2],models[3]],[1,1],[th1,th2],idlef,powerp)
            a=activetime_func([b,th1], models[3].act_popt[0], models[3].act_popt[1], models[3].act_popt[2], models[3].act_popt[3], models[3].act_popt[4])
            print(a)

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
    print(addinferece,addbatch,addresource)
    latency=predict(addinferece,addbatch,addresource,idlef,powerp)[1]
    li=0
    for mi in range(len(resource)):
        print(resource,slo)
        print(latency)
        if(resource[mi]>0):
            if(latency[li]>slo[mi]):
                ans[mi]=False
            li+=1
    print("tr",ans)
    return ans

def algorithm(inference,slo,arrivate,idlef,powerp):
    #inference: models
    m=len(inference)
    resource=[[0] * m for i in range(m)]
    batch=[0] * m
    r_lower=[[i,0] for i in range(m)]
    #print("batch",m,batch)
    # arrivate ips 
    arrivate=[i/1000 for i in arrivate]
    for i in range(m):
        inf=inference[i]
        #print("slo",slo[i],"ar",arrivate[i],"b",context.bandwidth,"data",inf.transferdata)
        batch[i]=math.ceil(slo[i]*arrivate[i]*context.bandwidth/(context.bandwidth+inf.transferdata*arrivate[i]))
        gama=inf.act_popt[0]*batch[i]**2+inf.act_popt[1]*batch[i]+inf.act_popt[2]
        delta=slo[i]-inf.baseidle-batch[i]*inf.transferdata/context.bandwidth-inf.act_popt[4]
        #print(gama-inf.act_popt[4]*delta,inf.act_popt[3]*delta*context.unit)
        r_lower[i][1]=math.ceil((gama/delta-inf.act_popt[3])/context.unit)*context.unit
    v=1
    print("batch",batch)
    print("r_lower",r_lower)
    r_lower.sort(key=lambda x:x[1], reverse=True)
    for ti in r_lower:
        i=ti[0]
        print(ti,"ti")
        inter=[100] * v
        ra_ij=copy.deepcopy(resource)
        for j in range(0,v):
            ra_j=0
            for inf in range(m):
                ra_j+=resource[inf][j]
            r_t=ra_j+ti[1]
            ra_ij[i][j]=ti[1]
            print("ra_j[j]",ra_j,"i",i,"j",j,"ra_ij",ra_ij,"resource",resource)
            print("r_t",r_t)
            while(r_t<=100):
                #tmpinferece=copy(inference)
                #tmpbatch=copy(batch)
                tmpresource=[tmp[j] for tmp in ra_ij]
                add=canadded(inference,batch,tmpresource,idlef,powerp,slo)
                c=True
                for d in add:
                    c=c and d
                print("c",c)
                if(c):
                    inter[j]=r_t-(ra_j+ti[1])
                    break
                else:
                    for tmp in range(m):
                        if(not add[tmp]):
                            ra_ij[tmp][j]+=context.unit
                            r_t+=context.unit
        mini=min(inter)
        j=inter.index(mini)
        print("jv",inter,v)
        if(mini<100):
            for inf in range(m):
                resource[inf][j]=ra_ij[inf][j]
        else:
            v+=1
            print(i,"r_lower",r_lower)
            resource[i][v-1]=ti[1]
        print("resource",resource)
    print("batch",batch)
    #print("resource",resource)
    lr=len(resource)
    for i in range(lr):
        print(i,slo[i],arrivate[i]*1000)
    for i in range(lr):
        print(i)
        ans=[]
        for j in range(lr):
            if(resource[j][i]!=0):
                ans.append([j,inference[j].name,batch[j],resource[j][i]])
        if(ans==[]):
            break
        print(ans)


def algorithm2(inference,slo,arrivate,idlef,powerp):
    #inference: models
    m=len(inference)
    resource=[[0] * m for i in range(m)]
    batch=[0] * m
    r_lower=[[i,0] for i in range(m)]
    #print("batch",m,batch)
    # arrivate ips 
    arrivate=[i/1000 for i in arrivate]
    for i in range(m):
        inf=inference[i]
        #print("slo",slo[i],"ar",arrivate[i],"b",context.bandwidth,"data",inf.transferdata)
        batch[i]=math.ceil(slo[i]*arrivate[i]*context.bandwidth/(context.bandwidth+inf.transferdata*arrivate[i]))
        gama=inf.act_popt[0]*batch[i]**2+inf.act_popt[1]*batch[i]+inf.act_popt[2]
        delta=slo[i]-inf.baseidle-batch[i]*inf.transferdata/context.bandwidth-inf.act_popt[4]
        #print(gama-inf.act_popt[4]*delta,inf.act_popt[3]*delta*context.unit)
        r_lower[i][1]=math.ceil((gama/delta-inf.act_popt[3])/context.unit)*context.unit
    v=1
    print("batch",batch)
    print("r_lower",r_lower)
    r_lower.sort(key=lambda x:x[1], reverse=True)
    for ti in r_lower:
        i=ti[0]
        c=False
        for j in range(v):
            ra_j=0
            for inf in range(m):
                ra_j+=resource[inf][j]
            if(ti[1]+ra_j<=100):
                resource[i][j]=ti[1]
                c=True
                break
        if(not c):
            v+=1
            resource[i][v-1]=ti[1]
        print("resource",resource)
    print("batch",batch)
    #print("resource",resource)
    lr=len(resource)
    for i in range(lr):
        print(i,slo[i],arrivate[i]*1000)
    for i in range(lr):
        print(i)
        ans=[]
        for j in range(lr):
            if(resource[j][i]!=0):
                ans.append([j,inference[j].name,batch[j],resource[j][i]])
        if(ans==[]):
            break
        print(ans)

        

if __name__ == '__main__':
    #profile=loadprofile("D:\\code\\config\\config")
    profile=loadprofile("./config")
    #p2=loadprofile("./gpulet.config")
    #solotest=loadprofile("./testcon_trans.log")
    contest=loadprofile("./testcon.log")
    context.bandwidth=profile["vgg19"]["htddur"][0]*profile["vgg19"]["transferdata"]/profile["vgg19"]["htddur"][1]
    context.idlepower=profile["hardware"]["idlepower"]
    powerp=power_frequency(profile["hardware"]["power"], profile["hardware"]["frequency"])
    idlef=idletime(profile["hardware"]["idletime"],profile["hardware"]["frequency"])

    model_par={}
    models=[]
    model2={}
    j=0
    for i in context.models:
        m=Model()
        m.name=i
        m.kernels=context.kernels[j]
        m.baseidle=profile[i]["idletime_1"]
        m.transferdata=profile[i]["transferdata"]
        j+=1
        m.l2cache=profile[i]["l2cache"]
        m.k_l2=((profile[i]["activetime_5"]*profile[i]["frequency_5"]/context.frequency)/profile[i]["activetime_1"]-1)/(4*m.l2cache)
        m.act_popt=activetime(profile[i]["gpulatency"],profile[i]["frequency"],profile[i]["idletime_1"])
        #m.act_popt=profile[i]["popt"]
        m.powers=solo_power(m.act_popt,profile[i]["power"],profile[i]["gpulatency"],profile[i]["idletime_1"],powerp)
        models.append(m)
        model2[i]=m
        #m.p()
    #gpuletco(idlef,powerp,model2,contest)
    #algorithm([models[0],models[0],models[1]],[6,8,10],[500,2000,600],idlef,powerp)
    #algorithm([models[0],models[0],models[0],models[0]],[5,10,15,20],[1000,400,300,200],idlef,powerp)
    algorithm2([models[0],models[0],models[0],models[1],models[1],models[1],\
        models[2],models[2],models[2],models[3],models[3],models[3]],\
            [5,7.5,10,5,10,15,10,15,20,15,20,25],[1000,500,2000,400,600,200,300,400,200,200,100,300],idlef,powerp)
    '''
    s=datetime.datetime.now()
    for i in range(1000):
        test(idlef,powerp,models)
    e=datetime.datetime.now()
    print(e-s)
    '''
    #accuracy of prediction
    #testpre(idlef,powerp,models)
    #testpre2(idlef,powerp,models)
    #nointerference()
    #analy(idlef,powerp,models)
    #yp = p_f(340,powerp[0])
