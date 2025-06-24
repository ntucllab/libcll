import os
import glob
import parse
import sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import scipy.stats as ss
import torch

key = "noise_real_0."
key = "uniform"
path = "logs"
datasets = ["mnist", "kmnist", "fmnist", "cifar10"]
datasets = ["mnist", "kmnist", "fmnist", "yeast", "texture", "control", "dermatology", "cifar10", "cifar20", "micro_imagenet10", "micro_imagenet20", "clcifar10", "clcifar20", "clmicro_imagenet10", "clmicro_imagenet20"]
datasets = ["mnist", "kmnist", "fmnist", "yeast", "texture", "control", "dermatology", "cifar10", "cifar20", "micro_imagenet10", "micro_imagenet20"]
datasets = ["clcifar10", "clcifar20", "clmicro_imagenet10", "clmicro_imagenet20"]
datasets = ["mnist", "kmnist", "fmnist", "cifar10", "micro_imagenet10"]
strategies = {"SCL-NL": {}, "SCL-EXP": {}, "SCL-FWD": {}, "URE-NN": {}, "URE-GA": {}, "DM": {}, "SCARCE": {}, "OP": {}, "PC": {}, "MCL-MAE": {}, "MCL-EXP": {}, "MCL-LOG": {}, "FWD": {}, "URE-TNN": {}, "URE-TGA": {}, "CPE-I": {}, "CPE-F": {}, "CPE-T": {}}
SS = [["PC", "SCL-NL", "SCL-EXP", "URE-NN", "URE-GA", "DM", "MCL-MAE", "MCL-EXP", "MCL-LOG", "OP", "SCARCE"], ["SCL-FWD", "URE-TNN", "URE-TGA", "CPE-I", "CPE-F", "CPE-T"]]
output_file = sys.argv[1]

keys = ["noisy_0.1", "noisy_0.2", "noisy_0.5"]
keys = ["noisy_0.1", "noisy_0.5"]
for key in keys:
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
        s = f"{s}-{tp}"
        v = int(s in SS[1])
        events = glob.glob(f"{file_name}/lightning_logs/version_{v}/*")
        acc = 0
        for event in events:
            event_acc = EventAccumulator(event)
            event_acc.Reload()
            if "Test_Accuracy" in event_acc.Tags()["scalars"]:
                acc = event_acc.Scalars("Test_Accuracy")[0].value
                break
        if res["dataset"] not in strategies[s]:
            strategies[s][res["dataset"]] = {k: {} for k in keys}
        if res["model"] == "MLP" or res["model"] == "ResNet34":
            strategies[s][res["dataset"]][key][seed] = acc * 100
print(strategies)
f = open(output_file, "w")
for dataset in datasets:
    for key in keys:
        for S in SS:
            mm = 0
            M = []
            for sss in S:
                m = sum(strategies[sss][dataset][key].values()) / len(strategies[sss][dataset][key])
                s = np.std(np.array(list(strategies[sss][dataset][key].values())))
                # if len(strategies[sss][dataset][key]) != 4:
                #     print("WWW")
                #     exit()
                print(sss, dataset, key, m, s)
                strategies[sss][dataset][key]["mean"] = m
                strategies[sss][dataset][key]["std"] = s
                mm =  max(mm, m)
                M.append(m)
            ranks = ss.rankdata(-np.array(M))
            M = torch.unique(torch.tensor(M), sorted=True)
            for i, sss in enumerate(S):
                format_m = "{:.2f}".format(strategies[sss][dataset][key]["mean"])
                format_s = "{:.2f}".format(strategies[sss][dataset][key]["std"])
                if strategies[sss][dataset][key]["mean"] == M[-1]:
                    strategies[sss][dataset][key]["str"] = f"\\textbf{{{format_m}}}\\scriptsize{{$\\pm${format_s}}}"
                elif strategies[sss][dataset][key]["mean"] == M[-2]:
                    strategies[sss][dataset][key]["str"] = f"\\underline{{{format_m}}}\\scriptsize{{$\\pm${format_s}}}"
                else:
                    strategies[sss][dataset][key]["str"] = f"{format_m}\\scriptsize{{$\\pm${format_s}}}"
                strategies[sss][dataset][key]["rank"] = ranks[i]

for key in keys:
    for S in SS:
        M = []
        for sss in S:
            strategies[sss][key] = sum([strategies[sss][dataset][key]["rank"] for dataset in datasets]) / len(datasets)
            M.append(strategies[sss][key])
        M = torch.unique(torch.tensor(M), sorted=True)
        for sss in S:
            format_r = "{:.2f}".format(strategies[sss][key])
            if strategies[sss][key] == M[0]:
                strategies[sss][key] = f"\\textbf{{{format_r}}}"
            elif strategies[sss][key] == M[1]:
                strategies[sss][key] = f"\\underline{{{format_r}}}"
            else:
                strategies[sss][key] = format_r

for S in SS:
    for sss in S:
        format_s = " & ".join([" & ".join([strategies[sss][dataset][key]["str"] for key in keys]) for dataset in datasets] + [" & ".join([strategies[sss][key] for key in keys])])
        print(sss + " & " + format_s + "\\\\", file=f)
    
# print(" ".join(lrs), file=f)
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
