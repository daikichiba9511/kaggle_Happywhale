""" this script is for ensemble


Reference
* https://www.kaggle.com/code/yamsam/simple-ensemble-of-public-best-kernels
"""
import csv
import pprint
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional

import pandas as pd

sub_files = [
    # "output/exp011-img736-all/fold0_submission.csv",
    # "output/exp011-img736-all/fold1_submission.csv",
    # "output/exp011-img736-all/fold2_submission.csv",
    # "output/exp011-img736-all/fold3_submission.csv",
    # "output/exp011-img736-all/fold4_submission.csv",
    # "./sub_ens_tf-eff07.csv",
    # "./sub_ens_convnext-large.csv",
    "output/exp012/fold0_submission.csv",
    "output/exp012/fold1_submission.csv",
    "output/exp012/fold2_submission.csv",
    "output/exp012/fold3_submission.csv",
    "output/exp012/fold4_submission.csv",
]


print("sub_files = \n ")
pprint.pprint(sub_files)


# sub_weights = [0.753, 0.755]
sub_weights = [
    0.80014030864,
    0.8070394868711886,
    0.800140308679094,
    0.8054319502906393,
    0.814126774115949,
]

print("cv mean score: ", sum(sub_weights) / len(sub_weights))
sub_weights = list(map(lambda x: x ** 2, sub_weights))


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
        for idx, target in enumerate(row1[h_target].split(" ")):
            target_weight[target] = target_weight.get(target, 0) + (place_wights[idx] * sub_weights[s])
    tops_target = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[h_label], " ".join(tops_target)])
out.close
