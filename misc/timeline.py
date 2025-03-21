import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime, date
from pprint import pprint
from random import shuffle

xls = pd.ExcelFile('./src/analysis/Overview algs in frameworks.xlsx')
df = pd.read_excel(xls, 'Properties')

frameworks = df['Property']
start = pd.to_datetime(df['First release'])
end = pd.to_datetime(df['Last updated'])
maintained = df['Maintained?']

data = list(zip(start,end,frameworks,maintained))
data = sorted(data,key = lambda x: (x[0],x[1]), reverse=True)


# Set the color of the grid lines
mpl.rcParams['grid.color'] = "w"

fig, ax = plt.subplots(1, 1)
# Plot eac item as a line
for i, (b, e, fw, m) in enumerate(data):
    
    if m == "yes" or m == "questionable":
        ax.plot_date([b,e], [i+1] * 2, ls='-', lw=10)
        ax.plot_date([e,date.today()], [i+1]*2, ls='-', lw=10,c=ax.get_lines()[-1].get_c(),alpha=0.5,marker=None)
    else:
        ax.plot_date([b,e], [i+1] * 2, ls='-', lw=10)  # 10 for the line width

_,_,fw,_ = zip(*data)

# Set ticks and labels on y axis
ax.set_yticks(range(1, len(fw) + 1))
ax.set_yticklabels(fw)

# Set color and transparency of the grid
ax.patch.set_facecolor('gray')
ax.patch.set_alpha(0.3)
# activate grid
ax.grid(True)
plt.show()