import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


xls = pd.ExcelFile('./src/analysis/PPO.xlsx')
df = pd.read_excel(xls, 'Blad1')

envs = df['Environment details']
envs = [x.split("-")[0] if "-" in x else x for x in envs]
envs = [x.replace("NoFrameskip","") for x in envs]

results = df['Results']
results = [str(x).replace("~","") for x in results]
results = [x.split(" ±")[0] if "±" in x else x for x in results]

entry = df['Entry']
frameworks = df['Framework']

assert(len(envs) == len(results))

res = {}

# transform to dic
for e,r,i in zip(envs,results,entry):
    if e != '':
        try:
            r = float(r)
            res[e] = res.get(e,[]) + [(r,i)]
        except:
            pass

# filter out environments with less than 3 entries
res = {x:y for x,y in zip(res.keys(),res.values()) if len(y) >= 3}


# calc z scores with theta = 1 (instead of 1.5)
theta = 1.0
Z_SCORES = {}
INDEXES_INFO = {}
IND = []

for i, key in enumerate(res.keys()):
    data = np.array([x[0] for x in res[key]])
    INDEXES_INFO[key] = np.array([x[1] for x in res[key]])

    IND = IND + [i+1]*len(data)

    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean)/std

    Z_SCORES[key] = z_scores

    # for i in np.where(np.abs(z_scores) > theta)[0]:
    #     print(f"{data[i]} in {key}")

# print("="*20)

outliers = []
for key in res.keys():
    data = np.array([x[0] for x in res[key]])
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - theta * IQR, Q3 + theta * IQR

    for i in np.where((data < lower_bound) | (data > upper_bound))[0]:
        ind = INDEXES_INFO[key][i]
        outliers.append((ind, data[i], key, frameworks[ind], ['over','under'][data[i] < lower_bound]))
        # print(f"{data[i]} in {key} from {ind}; framework = {frameworks[ind]}; {['over','under'][data[i] < lower_bound]}")

for i in sorted(outliers):
    print(i)

print("="*20)

for i in sorted([(x, len(Z_SCORES[x])) for x in Z_SCORES.keys()], key = lambda x: -x[1]):
    print(i)

def concat(x: dict):
    R = []
    for i in x.keys():
        R.extend(x[i])
    return R

fig,ax = plt.subplots()

Z_SCORES = list(Z_SCORES.items())
Z_SCORES.sort()
keys,vals = zip(*Z_SCORES)

ax.boxplot(vals,whis=theta)
# ax.scatter(IND, concat(Z_SCORES))
ax.set_xticklabels(keys,rotation=90)
plt.show()




