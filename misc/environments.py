import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


xls = pd.ExcelFile('./src/analysis/PPO.xlsx')
df = pd.read_excel(xls, 'Blad1')

envs = df['Environment details']
envs = [x.split("-")[0] if "-" in x else x for x in envs]
envs = [x.replace("NoFrameskip","") for x in envs]

entry = df['Entry']
frameworks = df['Framework']


print(['Environment details'])

datasb = df.loc[df['Framework'] == 'SB3']['Environment details']
datats = df.loc[df['Framework'] == 'Tianshou']['Environment details']
datatr = df.loc[df['Framework'] == 'TorchRL']['Environment details']



print(set(datasb).intersection(set(datats)).intersection(set(datatr)))