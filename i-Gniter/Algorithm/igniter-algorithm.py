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
    """
    This function loads profile data from a JSON file and returns it as a dictionary.
    :param path: A string representing the path to the JSON file.
    :return: A dictionary containing the loaded profile data.
    """
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


def p_f(po, a):
    return a * (po - context.power) + context.frequency


def power_frequency(power, frequency):
    """
    This function calculates the power frequency of a `VGG-19` model given the power and frequency data. The idle percent of the `VGG-19` model is very low and its GPU power grows linearly.
    :param power: A list of power values for the VGG-19 model.
    :param frequency: A list of frequency values for the VGG-19 model.
    :return: An array of optimal parameter values for the power frequency equation
    """
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
    """
    This function calculates the idle time based on the input data of idle time and frequency.
    :param idletime: A list of idle time data points.
    :param frequency: A list of frequency data points corresponding to the idle time data points.
    :return: An array of two fitted parameters using the curve_fit function based on the idle time data.
    """
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
    """
    This function calculates the active time for a GPU given the GPU latency, frequency, and base idle time.
    :param gpulatency: A 2D list of GPU latency for different batch sizes and threads.
    :param frequency: A 2D list of GPU frequency for different batch sizes and threads.
    :param baseidletime: The base idle time for the GPU.
    :return: The parameters obtained from the curve fitting.
    """
    activelatency = []
    x = [[], []]
    for i in range(len(context.batchsize)):
        for j in range(len(context.thread)):
            tmp = gpulatency[i][j]
            if frequency[i][j] < context.frequency:
                tmp *= (frequency[i][j] / context.frequency)
            tmp -= baseidletime
            activelatency.append(tmp)
            x[0].append(context.batchsize[i])
            x[1].append(context.thread[j])
    popt, pcov = curve_fit(activetime_func, np.array(x), np.array(activelatency), maxfev=100000)
    return popt


def ability_cp(x, k, b):
    return k * x + b


def power_l2cache(power, gpulatency, baseidletime, idlepower, l2caches, frequency):
    """
    This function calculates the optimal values of the parameters popt_p and popt_c for a given input data, using a non-linear curve fitting method.
    :param power: A 3x3 list of power values, where power[i][j] corresponds to the power consumed by i-th batch size and j-th thread.
    :param gpulatency: A 3x3 list of GPU latency values, where gpulatency[i][j] corresponds to the latency of i-th batch size and j-th thread.
    :param baseidletime: A float value representing the base idle time.
    :param idlepower: A float value representing the power consumed in idle state.
    :param l2caches: A 3x2 list of L2 cache values, where l2caches[i] corresponds to the L2 cache sizes of i-th batch size, i.e., [1, 10], [16, 50], and [32, 100].
    :param frequency: A 3x3 list of GPU frequency values, where frequency[i][j] corresponds to the frequency of i-th batch size and j-th thread.
    :return: popt_p: A tuple representing the optimal values of the parameters for power curve.
             popt_c: A tuple representing the optimal values of the parameters for L2 cache curve.
    """
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
    popt_p, pcov_p = curve_fit(ability_cp, ability_p, ypower)
    popt_c, pcov_c = curve_fit(ability_cp, [ability[0], ability[4], ability[8]],
                               l2caches)  # [[1, 10] [16, 50] [32, 100]]

    return popt_p, popt_c


def predict(models, batchsize, thread):
    """
    This function predicts the inference latency and throughput of each model based on the models placed on the current GPU.
    :param models: A list of Model objects containing details of the models such as idle time, activation function, power consumption, L2 cache, etc.
    :param batchsize: A list of integers representing batch size for each model.
    :param thread: A list of integers representing the number of threads used for each model.
    :return: A list containing two sub-lists. The first sub-list contains throughput for each model,and the second sub-list contains inference latency for each model.
    """
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


def durationps(openfilepath):
    """
    This function calculates the average duration and standard deviation of GPU and inference latency for a given log file.
    :param openfilepath: A string representing the path of the log file to be read.
    :return: A tuple containing two lists. The first list contains the mean duration for GPU and inference latency, respectively. The second list contains the standard deviation for GPU and inference latency, respectively.
             If the log file contains less than 10 valid entries, the function raises a "Duration time is too short!" warning message.
    """
    with open(openfilepath, encoding='utf-8') as f:
        log = f.readlines()
        gpulatency = []
        inferencelatency = []
        for i in range(1, len(log) - 1):
            j = eval(log[i][1:])
            if (j["startComputeMs"] > 10000 and j["endComputeMs"] < 20000):
                gpulatency.append(j["computeMs"])
                inferencelatency.append(j["latencyMs"])
        if len(gpulatency) > 10:
            return ([np.mean(gpulatency), np.mean(inferencelatency)], [np.std(gpulatency), np.std(inferencelatency)])
        else:
            print("Duration time is too short!")


def r_total(ra_j):
    ans = 0
    for r in ra_j:
        ans += r[1]
    return ans


def alloc_gpus(inference, batch, slo, ra_ij, r_lower, w, j):
    """
    Allocate GPUs for the given workloads based on the resources available and the Service Level Objectives (SLOs) of the workloads.
    :param inference: A list of inference models.
    :param batch: A list of batch sizes for each inference model.
    :param slo: A list of Service Level Objectives (SLOs) for each inference model.
    :param ra_ij: A nested list of tuples, where each tuple represents the workload and the allocated resources for that workload on a specific GPU.
    :param r_lower: The lower bound of resources that can be allocated to a workload
    :param w: The workload index to be allocated.
    :param j: The index of the GPU to allocate the workload to.
    :return: ra_ij: An updated list of allocated resources for each GPU after allocating the workload
    """
    flag = 1
    ra_ij[j].append([w, r_lower])
    if r_total(ra_ij[j]) > 100:  # If the sum of the resources of all models on that GPU exceeds 100
        return ra_ij
    addinferece = []
    addbatch = []
    addresource = []

    num_workloads = len(ra_ij[j])
    for infwork in ra_ij[j]:
        addinferece.append(inference[infwork[0]])
        addbatch.append(batch[infwork[0]])
        addresource.append(infwork[1])

    while np.sum(addresource) <= 100 and flag == 1:
        # After adding a load, the latency of other models may be predicted to change afterwards, so you have to keep
        # modifying until the latency of all models meet the requirements
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
    """
    This function determines how to place the DNN inference load and allocates GPU resources based on the user's input workload, latency, and request rate.
    :param inference: A list of models to allocate to GPUs.
    :param slo: A list of target SLOs (Service Level Objectives) for each model in the 'inference' list.
    :param rate: A list of inference request rates for each model in the 'inference' list.
    :return: None
    """
    for i in range(len(inference)):
        print(inference[i].name, slo[i], rate[i])
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
            # Give 100% resources, traverse batchsize
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
                print("GPU can't handle the load")
            else:
                new_rate = math.ceil(new_rate * 0.9)
                while rate[id] > new_rate:
                    result.append([id, inference[id].name, max_batch, 100, new_rate, slo[id]])
                    rate[id] -= new_rate

    g = 1  # Indicates the number of GPUs
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
    # Placement operation
    for ti in r_lower:
        i = ti[0]  # i denotes the i-th model
        r_l = ti[1]
        ra_ij = copy.deepcopy(resource)  # denotes the GPU resource assigned to the model i placed on the GPU j
        q = -1  # q denotes the GPU id that i the model is assigned to
        r_inter_min = 101  # Initialize minimal interference
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
    context.unit = profile["hardware"]["unit"]  # 100 / (SM / 2)
    context.bandwidth = profile["hardware"]["bandwidth"]  # PCIe
    context.kernels = []

    context.idlepower = profile["hardware"]["idlepower"]
    context.powerp = power_frequency(profile["hardware"]["power"], profile["hardware"]["frequency"])
    context.idlef = idletime(profile["hardware"]["idletime"], profile["hardware"]["frequency"])
    model_par = {}
    models = []
    model2 = {}
    j = 0
    # motivation example
    # model0: AlexNet, model1: ResNet-50, model2: VGG-19
    # workloads = [models[0], models[1], models[2]]
    # SLOs = [15, 40, 60]
    # rates = [500, 400, 200]

    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-m",
        "--model_slo_rate",
        type=str,
        action="append",
        help='Inputs include the model, latency constraints and request rates in this form [model:slo:rate]'
    )
    FLAGS = parse.parse_args()
    SLOs = [15, 40, 60]
    rates = [500, 400, 200]
    model_list = ["alexnet", "resnet50", "vgg19"]
    if FLAGS.model_slo_rate:
        print(FLAGS.model_slo_rate)
        SLOs = []
        rates = []
        model_list = []
        for model_config in FLAGS.model_slo_rate:
            model, slo, rate = model_config.split(":")
            model_list.append(model)
            SLOs.append(int(slo))
            rates.append(int(rate))

    for i in model_list:
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

    workloads = [models[i] for i in range(len(SLOs))]
    algorithm(workloads, SLOs, rates)
