import os
import glob
import parse
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

key = "noise_real_0."
key = "uniform"
path = "logs"
datasets = ["mnist", "kmnist", "fmnist", "cifar10", "micro_imagenet10"]
strategies = {"SCL-NL": {}, "SCL-EXP": {}, "SCL-FWD": {}, "URE-NN": {}, "URE-GA": {}, "DM-None": {}, "SCARCE-None": {}, "OP-None": {}, "PC-None": {}, "MCL-MAE": {}, "MCL-EXP": {}, "MCL-LOG": {}, "FWD-None": {}, "URE-TNN": {}, "URE-TGA": {}, "CPE-I": {}, "CPE-F": {}, "CPE-T": {}}
sss = sys.argv[1]
output_file = sys.argv[2]

keys = ["noisy_0.1", "noisy_0.2", "noisy_0.5"]
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
        if seed != "1126":
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
            strategies[s][res["dataset"]] = {k: {} for k in keys}
        if res["model"] == "MLP" or res["model"] == "ResNet34":
            strategies[s][res["dataset"]][key][lr] = acc * 100
print(strategies)
f = open(output_file, "w")
lrs = []
for dataset in datasets:
    for key in keys:
        l = "-1"
        acc = 0
        if len(strategies[sss][dataset][key]) != 5:
            print(sss, dataset)
        for lr in strategies[sss][dataset][key]:
            if strategies[sss][dataset][key][lr] > acc:
                l = lr
                acc = strategies[sss][dataset][key][lr]
        lrs.append(str(l))
print(" ".join(lrs), file=f)
