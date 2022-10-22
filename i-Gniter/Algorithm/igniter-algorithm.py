import argparse
import json
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


context = Context()


def loadprofile(path):
    profile = {}
    with open(path, "r") as file:
        for line in file:
            # print(line)
            data = json.loads(line)
            for key in data:
                if (key not in profile):
                    profile[key] = data[key]
                else:
                    profile[key].update(data[key])

    return profile


def p_f(po, a):
    # power to frequency when frequency>1530
    return a * (po - context.power) + context.frequency


def power_frequency(power, frequency):
    # the idle percent of VGG19 is very low, the GPU power of VGG-19 grows linearly.
    # p_f(power,popt[0])
    k = 0
    l = 0
    for i in range(len(frequency)):
        if (frequency[i] == context.frequency):
            if (i >= 1):
                k += (power[i] - power[i - 1])
                l += 1
        else:
            if l > 0:
                k /= l
            else:
                k = power[0] - context.idlepower
            for j in range(i, len(power)):
                power[j] = power[j - 1] + k
            popt, pcov = curve_fit(p_f, power[i:], frequency[i:])
            return popt


def idle_f(n_i, alpha_sch, beta_sch):
    return alpha_sch * n_i + beta_sch


def idletime(idletime, frequency):
    # use : idle_f(number of inference workloads, popt[0], popt[1])
    # vgg19
    l = len(idletime)
    vgg19 = [0] * l
    for i in range(l):
        vgg19[i] = idletime[i] * frequency[i] / context.frequency
    fp = []
    x = np.array([2, 3, 4, 5])
    for i in range(1, len(vgg19)):
        fp.append(vgg19[i] - vgg19[0])
    fp = np.array(fp)
    popt, pcov = curve_fit(idle_f, x, fp)
    return popt


def activetime_func(x, k1, k2, k3, k4, k5):
    r = (k1 * x[0] ** 2 + k2 * x[0] + k3) / (x[1] + k4) + k5
    return r


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


def ability_cp(x, k, b):
    return k * x + b


# l2caches 对应的应该是[b,r] =  [[1, 10] [16, 50] [32, 100]]
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
            if frequency[i][j] > context.frequency - 1:
                ability_p.append((1000 * b) / (gpulatency[i][j] - baseidletime))
                ypower.append(power[i][j] - idlepower)
    # print(ypower)
    popt_p, pcov_p = curve_fit(ability_cp, ability_p, ypower)
    popt_c, pcov_c = curve_fit(ability_cp, [ability[0], ability[4], ability[8]], l2caches) # [[1, 10] [16, 50] [32, 100]]

    return popt_p, popt_c


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
    if total_power > context.power:
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


def error(act, pre):
    err = []
    for i in range(len(act)):
        err.append(abs(act[i] - pre[i]) / act[i])
        # print(act[i],pre[i],err[i])
    return err


def durationps(openfilepath):
    with open(openfilepath, encoding='utf-8') as f:
        log = f.readlines()
        gpulatency = []
        inferencelatency = []
        for i in range(1, len(log) - 1):
            j = eval(log[i][1:])
            # j = json.loads(log[i])
            if (j["startComputeMs"] > 10000 and j["endComputeMs"] < 20000):
                # latencyMs computeMs
                gpulatency.append(j["computeMs"])
                inferencelatency.append(j["latencyMs"])
        if len(gpulatency) > 10:
            # print(sum/n,sum2/n)
            return ([np.mean(gpulatency), np.mean(inferencelatency)], [np.std(gpulatency), np.std(inferencelatency)])
        else:
            print("Duration time is too short!")


def r_total(ra_j):
    ans = 0
    for r in ra_j:
        ans += r[1]
    return ans


def alloc_gpus(inference, batch, slo, ra_ij, r_lower, w, j):
    flag = 1
    ra_ij[j].append([w, r_lower])
    if r_total(ra_ij[j]) > 100:  # 如果该GPU上的所有模型的资源之和超过了100
        return ra_ij
    addinferece = []
    addbatch = []
    addresource = []

    num_workloads = len(ra_ij[j])
    for infwork in ra_ij[j]:
        addinferece.append(inference[infwork[0]])
        addbatch.append(batch[infwork[0]])
        addresource.append(infwork[1])

    while np.sum(addresource) <= 100 and flag == 1:  # 增加一个负载之后，可能预测之后其他模型的latency发生变化，所以就要一直修改，直到所有模型的latency都满足要求
        flag = 0
        latency = predict(addinferece, addbatch, addresource)[1]
        for i in range(num_workloads):
            if latency[i] > slo[ra_ij[j][i][0]]:
                addresource[i] += context.unit
                flag = 1

    for i in range(num_workloads):
        ra_ij[j][i][1] = addresource[i]
    return ra_ij


def algorithm(inference, slo, rate):
    # inference: models

    # Initialize:
    slo = [i / 2 for i in slo]
    m = len(inference)
    resource = [[]]
    batch = [0] * m
    r_lower = [[i, 0] for i in range(m)]
    # arrivate ips
    arrivate = [i / 1000 for i in rate]
    for i in range(m):
        inf = inference[i]
        batch[i] = math.ceil(
            slo[i] * arrivate[i] * context.bandwidth / (context.bandwidth + inf.inputdata * arrivate[i]))
        gama = inf.act_popt[0] * batch[i] ** 2 + inf.act_popt[1] * batch[i] + inf.act_popt[2]
        delta = slo[i] - inf.baseidle - batch[i] * (inf.inputdata + inf.outputdata) / context.bandwidth - inf.act_popt[
            4]
        r_lower[i][1] = math.ceil((gama / delta - inf.act_popt[3]) / context.unit) * context.unit
    # Initialize End

    result = []
    for i in range(len(r_lower)):
        if r_lower[i][1] > 100:
            print("workload is too large!")
            # 给100%资源，遍历batchsize
            max_batch = -1
            new_rate = 0
            id = r_lower[i][0]
            for b in range(1, 33):
                res = predict([inference[id]], [b], [100])
                if res[1][0] < slo[id]:
                    max_batch = b
                    new_rate = res[0][0]
                else:
                    break
            if max_batch == -1:
                print("GPU 无法满足该负载")
            else:
                new_rate = math.ceil(new_rate * 0.9)
                while rate[id] > new_rate:
                    result.append([id, inference[id].name, max_batch, 100, new_rate, slo[id]])
                    rate[id] -= new_rate


    g = 1  # 表示GPU的数量
    arrivate = [i / 1000 for i in rate]
    for i in range(m):
        inf = inference[i]
        batch[i] = math.ceil(
            slo[i] * arrivate[i] * context.bandwidth / (context.bandwidth + inf.inputdata * arrivate[i]))
        gama = inf.act_popt[0] * batch[i] ** 2 + inf.act_popt[1] * batch[i] + inf.act_popt[2]
        delta = slo[i] - inf.baseidle - batch[i] * (inf.inputdata + inf.outputdata) / context.bandwidth - inf.act_popt[
            4]
        r_lower[i][1] = math.ceil((gama / delta - inf.act_popt[3]) / context.unit) * context.unit

    # sorted
    r_lower.sort(key=lambda x: x[1], reverse=True)
    # 放置操作
    for ti in r_lower:
        i = ti[0]  # i表示第i个模型
        r_l = ti[1]
        ra_ij = copy.deepcopy(resource)  # 表示把i这个模型放置到j这个GPU上被分配的GPU resource
        q = -1  # q表示i这个模型被分配到的GPU id
        r_inter_min = 101  # 最小的干扰
        for j in range(g):
            ra_ij = alloc_gpus(inference, batch, slo, ra_ij, r_l, i, j)
            r_inter = r_total(ra_ij[j]) - r_total(resource[j])
            if r_total(ra_ij[j]) <= 100 and r_inter < r_inter_min:
                q = j
                r_inter_min = r_inter
        if q == -1:
            g += 1
            resource.append([])
            resource[g - 1].append([i, r_l])
        else:
            resource[q] = copy.deepcopy(ra_ij[q])

    print(resource)
    gpuid = 1
    for gpu in resource:
        ans = []
        conf = {"models": [], "rates": [], "slos": [], "resources": [], "batches": []}
        for inf in gpu:
            ans.append([inf[0], inference[inf[0]].name, batch[inf[0]], inf[1]])
            conf["models"].append(inference[inf[0]].name)
            conf["rates"].append(rate[inf[0]])
            conf["slos"].append(slo[inf[0]])
            conf["resources"].append(inf[1])
            conf["batches"].append(batch[inf[0]])
        if (ans == []):
            break

        print("GPU", gpuid)
        print(ans)
        with open("./config_gpu" + str(gpuid) + ".json", "w") as f:
            json.dump(conf, f)
        gpuid += 1

    # add additional gpu config
    for model in result:
        conf = {"models": [], "rates": [], "slos": [], "resources": [], "batches": []}
        conf["models"].append(model[1])
        conf["rates"].append(model[4])
        conf["slos"].append(model[5])
        conf["resources"].append(model[3])
        conf["batches"].append(batch[model[2]])

        with open("./config_gpu" + str(gpuid) + ".json", "w") as f:
            json.dump(conf, f)
        gpuid += 1

if __name__ == '__main__':
    profile = loadprofile("./config")

    context.frequency = profile["hardware"]["maxGPUfrequency"]
    context.power = profile["hardware"]["maxGPUpower"]
    context.unit = profile["hardware"]["unit"] # 100 / (SM / 2)
    context.bandwidth = profile["hardware"]["bandwidth"] # PCIe
    context.kernels = []


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
        m.kernels = profile[i]["kernel"]
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
        context.kernels.append(m.kernels)
        models.append(m)
        model2[i] = m

    # motivation example
    # model0: AlexNet, model1: ResNet-50, model2: VGG-19
    # workloads = [models[0], models[1], models[2]]
    # SLOs = [15, 40, 60]
    # rates = [500, 400, 200]

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
    SLOs = [15, 40, 60]
    rates = [500, 400, 200]
    if FLAGS.slos:
        SLOs = [int(x) for x in FLAGS.slos.split(":")]
    if FLAGS.rates:
        rates = [int(x) for x in FLAGS.rates.split(":")]
    workloads = [models[i] for i in range(len(SLOs))]
    algorithm(workloads, SLOs, rates)
