""" this script is for ensemble


Reference
* https://www.kaggle.com/code/yamsam/simple-ensemble-of-public-best-kernels
"""
import csv
import pandas as pd
import pprint

from typing import Optional, List, Dict, Any


sub_files = [
    "output/exp000/convnext-large-in22k-makesub/fold0-submission.csv",
    "output/exp000/convnext-large-in22k-makesub/fold1-submission.csv",
    "output/exp000/convnext-large-in22k-makesub/fold2-submission.csv",
    "output/exp000/convnext-large-in22k-makesub/fold3-submission.csv",
    "output/exp000/convnext-large-in22k-makesub/fold4-submission.csv",
]


print("sub_files = \n ")
pprint.pprint(sub_files)


sub_weights = [
    (10.0 - 9.544) ** 2,
    (10.0 - 9.570) ** 2,
    (10.0 - 9.563) ** 2,
    (10.0 - 9.610) ** 2,
    (10.0 - 9.596) ** 2,
]


h_label = "image"
h_target = "predictions"

npt = 6
place_wights = {i: 1 / (i + 1) for i in range(npt)}
print(f"place_wights = {place_wights}")

num_sub_files = len(sub_files)
sub = [None] * num_sub_files

for i, file in enumerate(sub_files):
    print(f"Reading {i}: w = {sub_weights[i]} -> {file}")
    reader = csv.DictReader(open(file, "r"))
    sub[i] = sorted(reader, key=lambda d: str(d[h_label]))

out = open("sub_ens.csv", "w", newline="")
writer = csv.writer(out)
writer.writerow([h_label, h_target])

for p, row in enumerate(sub[0]):
    target_weight: Dict[str, Any] = {}
    for s in range(num_sub_files):
        row1 = sub[s][p]
        for idx, target in enumerate(row[h_target].split(" ")):
            target_weight[target] = target_weight.get(target, 0) + (
                place_wights[idx] * sub_weights[s]
            )
    tops_target = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[h_label], " ".join(tops_target)])
out.close
