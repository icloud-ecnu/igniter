# coding=utf-8
import argparse
import json
import os

import numpy as np
from scipy.optimize import curve_fit
import math
import copy


class Context:
    frequency = 1530
    power = 300
    idlepower = 52
    batchsize = [1, 16, 32]
    thread = [10, 50, 100]
    models = ["alexnet", "resnet50", "vgg19", "ssd"]
    kernels = [20, 80, 29, 93]
    unit = 2.5
    step = 40
    bandwidth = 10
    powerp = []
    idlef = []


class Model:
    name = "alexnet"
    kernels = 29
    baseidle = 0.1
    # act_popt[0-4] k1-k5
    act_popt = []
    l2cache = 0
    power_popt = []
    l2cache_popt = []
    k_l2 = 1
    tmpl2cache = 0
    inputdata = 1000
    outputdata = 1000


class ModelStatus:
    model = Model()
    batch = 1
    resource = 0
    slo = 0
    rate = 0


context = Context()


def idle_f(n_i, alpha_sch, beta_sch):
    return alpha_sch * n_i + beta_sch


def activetime_func(x, k1, k2, k3, k4, k5):
    r = (k1 * x[0] ** 2 + k2 * x[0] + k3) / (x[1] + k4) + k5
    return r


def ability_cp(x, k, b):
    return k * x + b


def p_f(po, a):
    # power to frequency when frequency>1530
    return a * (po - context.power) + context.frequency


def predict(models, batchsize, thread):  # model类型是Model类,batch,resource
    # batchsize [[throughput] ,[inference latency]]
    p = len(models)
    ans = [[], []]
    increasing_idle = 0
    if p >= 2:
        increasing_idle = idle_f(p, context.idlef[0], context.idlef[1])
    total_l2cache = 0
    total_power = context.idlepower
    frequency = context.frequency
    tmpcache = []
    for i in range(len(models)):
        m = models[i]
        tact_solo = activetime_func([batchsize[i], thread[i]], m.act_popt[0], m.act_popt[1], m.act_popt[2],
                                    m.act_popt[3], m.act_popt[4])
        tmppower = ability_cp((1000 * batchsize[i]) / tact_solo, m.power_popt[0], m.power_popt[1])
        total_power += tmppower
        tmpl2cache = ability_cp((1000 * batchsize[i]) / tact_solo, m.l2cache_popt[0], m.l2cache_popt[1])
        tmpcache.append(tmpl2cache)
        total_l2cache += tmpl2cache
    if total_power > 300:
        frequency = p_f(total_power, context.powerp[0])
    for i in range(len(models)):
        m = models[i]
        idletime = m.kernels * increasing_idle + m.baseidle
        tact_solo = activetime_func([batchsize[i], thread[i]], m.act_popt[0], m.act_popt[1], m.act_popt[2],
                                    m.act_popt[3], m.act_popt[4])
        l2 = total_l2cache - tmpcache[i]
        activetime = tact_solo * (1 + m.k_l2 * l2)
        gpu_latency = (idletime + activetime) / (frequency / context.frequency)

        ans[0].append(
            batchsize[i] * 1000 / (gpu_latency + m.outputdata * batchsize[i] / context.bandwidth))  # throughput
        ans[1].append(gpu_latency + (m.inputdata + m.outputdata) * batchsize[
            i] / context.bandwidth)  # T(inf) = T(gpu) + T(load) + T(feedback)

    return ans


def activetime(gpulatency, frequency, baseidletime):
    # activetime_func([b,th], popt[0], popt[1], popt[2], popt[3], popt[4])
    activelatency = []
    x = [[], []]
    # batchsize thread
    for i in range(len(context.batchsize)):
        for j in range(len(context.thread)):
            tmp = gpulatency[i][j]
            if (frequency[i][j] < context.frequency):
                tmp *= (frequency[i][j] / context.frequency)
            tmp -= baseidletime
            activelatency.append(tmp)
            x[0].append(context.batchsize[i])
            x[1].append(context.thread[j])
    popt, pcov = curve_fit(activetime_func, np.array(x), np.array(activelatency), maxfev=100000)
    return popt


def loadprofile(path):
    profile = {}
    with open(path, "r") as file:
        for line in file:
            data = json.loads(line)
            for key in data:
                if (key not in profile):
                    profile[key] = data[key]
                else:
                    profile[key].update(data[key])

    return profile


def power_frequency(power, frequency):
    # the idle percent of VGG19 is very low, the GPU power of VGG-19 grows linearly.
    # p_f(power,popt[0])
    k = 0
    l = 0
    for i in range(len(frequency)):
        if (frequency[i] == 1530):
            if (i >= 1):
                k += (power[i] - power[i - 1])
                l += 1
        else:
            k /= l
            for j in range(i, len(power)):
                power[j] = power[j - 1] + k
            popt, pcov = curve_fit(p_f, power[i:], frequency[i:])
            return popt


def idletime(idletime, frequency):
    # use : idle_f(number of inference workloads, popt[0], popt[1])
    # vgg19
    l = len(idletime)
    vgg19 = [0] * l
    for i in range(l):
        vgg19[i] = idletime[i] * frequency[i] / 1530
    fp = []
    x = np.array([2, 3, 4, 5])
    for i in range(1, len(vgg19)):
        fp.append(vgg19[i] - vgg19[0])
    fp = np.array(fp)
    popt, pcov = curve_fit(idle_f, x, fp)
    return popt


def power_l2cache(power, gpulatency, baseidletime, idlepower, l2caches, frequency):
    # power gpulatecny t50
    # power[batchsize][thread]
    # popt act_popt
    batch = [1, 16, 32]
    resource = [10, 50, 100]
    ability = []
    ability_p = []
    ypower = []
    for i in range(3):
        for j in range(3):
            b = batch[i]
            r = resource[j]
            ability.append((1000 * b) / (gpulatency[i][j] - baseidletime))
            if frequency[i][j] > 1529:
                ability_p.append((1000 * b) / (gpulatency[i][j] - baseidletime))
                ypower.append(power[i][j] - idlepower)
    popt_p, pcov_p = curve_fit(ability_cp, ability_p, ypower)
    popt_c, pcov_c = curve_fit(ability_cp, [ability[0], ability[4], ability[8]], l2caches)

    return popt_p, popt_c


def total_GPU_Resource(gpu_status):
    ans = 0
    for modelStatus in gpu_status:
        ans += modelStatus.resource
    return ans


# check 用来判断当前模型的状态能否放到gpu_status上
def check(model, batch, resource, gpu_status, slo, rate):
    total_gpu_resource = total_GPU_Resource(gpu_status)
    if total_gpu_resource + resource > 100:
        return False, gpu_status

    addInference = [model]
    addBatch = [batch]
    addResource = [resource]
    addSlo = [slo]
    addRate = [rate]
    for modelStatus in gpu_status:
        addInference.append(modelStatus.model)
        addBatch.append(modelStatus.batch)
        addResource.append(modelStatus.resource)
        addSlo.append(modelStatus.slo)
        addRate.append(modelStatus.rate)

    throughput, latency = predict(addInference, addBatch, addResource)
    for i in range(len(addInference)):
        if throughput[i] < addRate[i]:
            return False, gpu_status
        if latency[i] > addSlo[i]:
            return False, gpu_status

    ans = copy.deepcopy(gpu_status)
    modelStatus = ModelStatus()
    modelStatus.model = model
    modelStatus.batch = batch
    modelStatus.resource = resource
    modelStatus.slo = slo
    ans.append(modelStatus)
    return True, ans


min_gpu_number = 100000000
best_gpu_allocate = []
fileName = 'force_algorithm_config.txt'


def printGPUAllocation(best_gpu_allocate):
    with open(fileName, 'a', encoding='utf-8') as file:
        file.write("min_gpu_number = {}\n".format(min_gpu_number))
        file.write("best_gpu_allocate:\n")
        for i in range(len(best_gpu_allocate)):
            file.write("第 {} 个GPU：\n".format(i))
            file.write("total_gpu_resource = {}\n".format(total_GPU_Resource(best_gpu_allocate[i])))
            for j in range(len(best_gpu_allocate[i])):
                modelStatus = best_gpu_allocate[i][j]
                file.write("modelName = {} batch = {} resources = {}\n".format(
                    modelStatus.model.name, modelStatus.batch, modelStatus.resource))
        file.write("\n")


def GPU_resource(gpu_status):
    ans = 0
    for i in range(len(gpu_status)):
        ans += total_GPU_Resource(gpu_status[i])
    return ans


def dfsAns(pos, gpu_status, models, slos, rates):
    """
    :param pos:表示的当前是第几个模型
    :param gpu_status:表示当前已经被分配的GPU的状态，包括：GPU上共置的模型和其配置，
           二维数组，[[ModelStatus,ModelStatus],[ModelStatus,ModelStatus]]
    :param batch:表示当前模型选择的batch
    :param resource:表示当前模型选择的GPU资源
    :param models:模型的列表
    :param slos: 模型的延迟约束
    :param rates: 模型的请求率
    :return:void
    """
    global min_gpu_number, best_gpu_allocate
    # 优化条件：
    if len(gpu_status) > min_gpu_number:
        return
    if min_gpu_number == len(gpu_status) and GPU_resource(best_gpu_allocate) <= GPU_resource(gpu_status):
        return

    # 终止条件，pos == len(models)
    if pos == len(models):
        if min_gpu_number > len(gpu_status):
            min_gpu_number = len(gpu_status)
            best_gpu_allocate = gpu_status
        elif min_gpu_number == len(gpu_status) and GPU_resource(best_gpu_allocate) > GPU_resource(gpu_status):
            best_gpu_allocate = gpu_status
            # printGPUAllocation(best_gpu_allocate)
        return

    for batch in range(1, 33):
        for coefficient in range(1, 41):  # [1,40] 40 = 100/2.5
            resource = coefficient * context.unit
            for gpu_index in range(len(gpu_status)):
                flag, tmp = check(models[pos], batch, resource, gpu_status[gpu_index],
                                  slos[pos], rates[pos])  # 如果该模型可以放在第gpu_index这个GPU上，则继续dfs下一个模型
                if flag:
                    new_gpu_status = copy.deepcopy(gpu_status)
                    new_gpu_status[gpu_index] = tmp
                    dfsAns(pos + 1, new_gpu_status, models, slos, rates)

            new_gpu_status = copy.deepcopy(gpu_status)
            flag, tmp = check(models[pos], batch, resource, [], slos[pos], rates[pos])
            if flag:
                new_gpu_status.append(tmp)
                dfsAns(pos + 1, new_gpu_status, models, slos, rates)


"""
希望找到一个最优解，最优解是在满足slo前提下最少使用的GPU数量，只需要考虑GPU数量，还是有其他指标
存在的问题：
是否需要判断：predict 预测出来负载的吞吐量 >= 负载的请求率？
论文中求解出来的batch是最优的吗？还是贪心选取的？还是最少需要的？

暴力求最优？
但是用了predict这个预测模型，所以求出来的不一定是最优
"""

if __name__ == '__main__':
    profile = loadprofile("./config")

    context.bandwidth = 10000000

    context.idlepower = profile["hardware"]["idlepower"]
    context.powerp = power_frequency(profile["hardware"]["power"], profile["hardware"]["frequency"])
    context.idlef = idletime(profile["hardware"]["idletime"], profile["hardware"]["frequency"])
    model_par = {}
    models = []
    model2 = {}
    j = 0
    for i in context.models:
        m = Model()
        m.name = i + "_dynamic"
        m.kernels = context.kernels[j]
        m.baseidle = profile[i]["idletime_1"]
        m.inputdata = profile[i]["inputdata"]
        m.outputdata = profile[i]["outputdata"]
        j += 1
        m.l2cache = profile[i]["l2cache"]
        m.k_l2 = ((profile[i]["activetime_2"] * profile[i]["frequency_2"] / context.frequency) / profile[i][
            "activetime_1"] - 1) / (m.l2cache)
        m.act_popt = activetime(profile[i]["gpulatency"], profile[i]["frequency"], profile[i]["idletime_1"])
        m.power_popt, m.l2cache_popt = power_l2cache(profile[i]["power"], profile[i]["gpulatency"], m.baseidle,
                                                     context.idlepower, profile[i]["l2caches"], profile[i]["frequency"])
        models.append(m)
        model2[i] = m

    # motivation example
    # model0: AlexNet, model1: ResNet-50, model2: VGG-19
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-s",
        "--slos",
        type=str,
    )
    parse.add_argument(
        "-r",
        "--rates",
        type=str,
    )
    FLAGS = parse.parse_args()
    workloads = [models[0], models[1], models[2]]
    SLOs = [15, 40, 60]
    rates = [500, 400, 200]
    # SLOs = [int(x) for x in FLAGS.slos.split(":")]
    # rates = [int(x) for x in FLAGS.rates.split(":")]
    # workloads = [models[0], models[1], models[2], models[3]]
    with open(fileName, 'a', encoding='utf-8') as file:
        file.write("\n=======模型配置============\n")
        for i in range(len(SLOs)):
            file.write("modelName = {} SLOS = {} rates = {}\n".format(workloads[i].name, SLOs[i], rates[i]))
        file.write("=====配置方案=====\n")
    dfsAns(0, [], workloads, SLOs, rates)
    printGPUAllocation(best_gpu_allocate)
