import os
import glob
import parse
import sys
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from autorank import autorank, plot_stats
import Orange
import scipy.stats as ss
import torch

key = "noise_real_0."
key = "uniform"
key = "multi_label_1-uniform"
path = "logs"
datasets = ["mnist", "kmnist", "fmnist", "cifar10"]
datasets = ["mnist", "kmnist", "fmnist", "yeast", "texture", "control", "dermatology", "cifar10", "cifar20", "micro_imagenet10", "micro_imagenet20", "clcifar10", "clcifar20", "clmicro_imagenet10", "clmicro_imagenet20"]
datasets = ["mnist", "kmnist", "fmnist", "yeast", "texture", "dermatology", "control", "cifar10", "cifar20", "micro_imagenet10", "micro_imagenet20"]
datasets = ["mnist", "kmnist", "fmnist", "cifar10", "cifar20", "micro_imagenet10", "micro_imagenet20"]
datasets = ["yeast", "texture", "dermatology", "control"]
datasets = ["clcifar10", "clcifar20", "clmicro_imagenet10", "clmicro_imagenet20"]
strategies = {"SCL-NL": {}, "SCL-EXP": {}, "SCL-FWD": {}, "URE-NN": {}, "URE-GA": {}, "DM": {}, "SCARCE": {}, "OP": {}, "PC": {}, "MCL-MAE": {}, "MCL-EXP": {}, "MCL-LOG": {}, "FWD": {}, "URE-TNN": {}, "URE-TGA": {}, "CPE-I": {}, "CPE-F": {}, "CPE-T": {}}
SS = [["PC", "SCL-NL", "SCL-EXP", "URE-NN", "URE-GA", "DM", "MCL-MAE", "MCL-EXP", "MCL-LOG", "OP", "SCARCE"], ["SCL-FWD", "URE-TNN", "URE-TGA", "CPE-I", "CPE-F", "CPE-T"]]
output_file = sys.argv[1]

file_names = glob.glob(f"{path}/*/*{key}*/*")

for file_name in file_names:
    format_string = "{strategy}-{tp}-{model}-{dataset}-{lr}"
    res = parse.parse(format_string, os.path.basename(file_name))
    if res["model"] != "MLP" and res["model"] != "ResNet34":
        continue
    s = res["strategy"]
    tp = res["tp"]
    lr = "-".join(res["lr"].split("-")[:2])
    seed = res["lr"].split("-")[-1]
    # print(os.path.basename(file_name))
    if seed == "1126":
        continue
    events = glob.glob(f"{file_name}/lightning_logs/version_0/*")
    acc = 0
    for event in events:
        event_acc = EventAccumulator(event)
        event_acc.Reload()
        if "Test_Accuracy" in event_acc.Tags()["scalars"]:
            acc = event_acc.Scalars("Test_Accuracy")[0].value
            break
    s = f"{s}-{tp}"
    if res["dataset"] not in strategies[s]:
        strategies[s][res["dataset"]] = {}
    if res["model"] == "MLP" or res["model"] == "ResNet34":
        strategies[s][res["dataset"]][seed] = acc * 100
print(strategies)
f = open(output_file, "w")
for dataset in datasets:
    for S in SS:
        M = []
        mm = 0
        for i, sss in enumerate(S):
            m = sum(strategies[sss][dataset].values()) / len(strategies[sss][dataset])
            s = np.std(np.array(list(strategies[sss][dataset].values())))
            if len(strategies[sss][dataset]) != 4:
                print("WWW")
            print(sss, dataset, m, s)
            strategies[sss][dataset]["mean"] = m
            strategies[sss][dataset]["std"] = s
            mm = max(mm, m)
            M.append(m)
        ranks = ss.rankdata(-np.array(M))
        M = torch.unique(torch.tensor(M), sorted=True)
        for i, sss in enumerate(S):
            format_m = "{:.2f}".format(strategies[sss][dataset]["mean"])
            format_s = "{:.2f}".format(strategies[sss][dataset]["std"])
            if strategies[sss][dataset]["mean"] == M[-1]:
                strategies[sss][dataset]["str"] = f"\\textbf{{{format_m}}}\\scriptsize{{$\\pm${format_s}}}"
            elif strategies[sss][dataset]["mean"] == M[-2]:
                strategies[sss][dataset]["str"] = f"\\underline{{{format_m}}}\\scriptsize{{$\\pm${format_s}}}"
            else:
                strategies[sss][dataset]["str"] = f"{format_m}\\scriptsize{{$\\pm${format_s}}}"
            strategies[sss][dataset]["rank"] = ranks[i]

for S in SS:
    M = []
    for sss in S:
        strategies[sss]["rank"] = sum([strategies[sss][dataset]["rank"] for dataset in datasets]) / len(datasets)
        M.append(strategies[sss]["rank"])
    M = torch.unique(torch.tensor(M), sorted=True)
    for sss in S:
        format_r = "{:.2f}".format(strategies[sss]["rank"])
        if strategies[sss]["rank"] == M[0]:
            strategies[sss]["rank"] = f"\\textbf{{{format_r}}}"
        elif strategies[sss]["rank"] == M[1]:
            strategies[sss]["rank"] = f"\\underline{{{format_r}}}"
        else:
            strategies[sss]["rank"] = format_r
    for sss in S:
        format_s = " & ".join([strategies[sss][dataset]["str"] for dataset in datasets] + [strategies[sss]["rank"]])
        print(sss + " & " + format_s + "\\\\", file=f)
    
# print(" ".join(lrs), file=f)
# df = pd.DataFrame(M, columns=SS[0])
# result = autorank(df, alpha=0.05, verbose=False, order='ascending')
# plot_stats(result)
# plt.savefig("test.png")
# plt.cla()

# N = np.array(M)
# # ranks = (-N).argsort(axis=-1).argsort(axis=-1) + 1
# ranks = ss.rankdata(-N)
# print(N, ranks)
# exit()
# avranks = ranks.mean()
# cd = Orange.evaluation.compute_CD(avranks, len(datasets)) #tested on 30 datasets
# Orange.evaluation.graph_ranks(avranks, SS[0], cd=cd, width=6, textspace=1.5)
# plt.savefig("test2.png")
# plt.cla()
exit()
f = open(output_file, "w")
for strategy in strategies:
    for method in strategies[strategy]:
        f.write(f"{strategy}-{method},")
        print(strategy, method)
        for dataset in datasets:
            if strategies[strategy][method] == {}:
                break
            print(dataset, dict(sorted(strategies[strategy][method][dataset].items())))
            acc = ",".join(map(str, list(dict(sorted(strategies[strategy][method][dataset].items())).values())))
            f.write(f"{acc},")
        f.write(f"\n")
