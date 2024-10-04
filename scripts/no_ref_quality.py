import pyiqa
import torch
import os
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, help='Name of the No Reference metric to use. Metrics used in the paper [niqe,nrqm,pi]')
parser.add_argument('--imgs_path', type=str, help='The path to the output image. For single image inference only.')

metric = parser["metric"]
folder = parser["imgs_path"]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create metric with default setting
niqe_metric = pyiqa.create_metric(metric, device=device)

results = []
for fi in tqdm(os.listdir(folder)):
    results.append(niqe_metric(f"{folder}/{fi}").detach().cpu().numpy())
result = sum(results) / len(results)
print(result)