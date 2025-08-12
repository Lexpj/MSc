import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime, date
from pprint import pprint
from random import shuffle
from matplotlib.patches import Circle

xls = pd.ExcelFile('./misc/analysis/Overview algs in frameworks.xlsx')
df = pd.read_excel(xls, 'Properties')

frameworks = df['Property']

start = pd.to_datetime(df['First release'])
end = pd.to_datetime(df['Last updated'])
maintained = df['Maintained?']
backend = df['Backend']

data = list(zip(start,end,frameworks,maintained,backend))
data = sorted(data,key = lambda x: (x[0],x[1]), reverse=True)

set1 = set([x[2] for x in data if 'TF' in x[4]]) - set(["Coach"])
set2 = set([x[2] for x in data if 'PyTorch' in x[4]])
set3 = set([x[2] for x in data if 'JAX' in x[4]])

venn = venn3([set1, set2, set3], ('TF', 'PyTorch', 'JAX'))
venn.get_label_by_id('100').set_text('\n'.join(set1-set2-set3))
venn.get_label_by_id('110').set_text('\n'.join(set1&set2-set3))
venn.get_label_by_id('010').set_text('\n'.join(set2-set3-set1))
venn.get_label_by_id('101').set_text('\n'.join(set1&set3-set2))
venn.get_label_by_id('111').set_text('\n'.join(set1&set2&set3))
venn.get_label_by_id('011').set_text('\n'.join(set2&set3-set1))
venn.get_label_by_id('001').set_text('\n'.join(set3-set2-set1))
# Add two unconnected circles with text

ax = plt.gca()
circle1 = Circle((-0.6, -0.2), 0.15, color='lightblue', alpha=0.5 )
circle2 = Circle((0.65, -0.4), 0.10, color='purple', alpha=0.5)
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.text(-0.55, -0.15, 'Coach', ha='center', va='center', fontsize=10)
ax.text(-0.75, -0.35, 'MXNet', ha='center', va='center', fontsize=12)

ax.text(0.65, -0.4, 'ChainerRL', ha='center', va='center', fontsize=10)
ax.text(0.5, -0.5, 'Chainer', ha='center', va='center', fontsize=12)
plt.show()