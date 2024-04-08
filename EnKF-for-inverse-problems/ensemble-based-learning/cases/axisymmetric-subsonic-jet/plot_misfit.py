import numpy as np
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# set plot properties
params = {
    'text.latex.preamble': '\\usepackage{gensymb}',
    'text.usetex': True,
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'figure.autolayout': True,
    'image.cmap': 'magma',
    'axes.grid': False,
    'savefig.dpi': 600,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'font.size': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': [10, 5],
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'lines.linewidth': 2,
    'lines.markersize': 6,
}
mpl.rcParams.update(params)


dafiLogFile = open('dafi.log')
allData = dafiLogFile.read()
dafiLogFile.close()

pattern=re.compile('misfit = ... (.+)s')
misFits=pattern.findall(allData)
misFits = np.array(misFits)

misFitsValue = misFits.astype(float)
NUM = len(misFitsValue)
orderNum=np.arange(1,NUM+1,1)

caseName = 'ASJet'
line_truth = ['-', 'tab:red', 2, (None, None), 1.0]
plt.plot(orderNum,misFitsValue,linestyle=line_truth[0],color=line_truth[1],lw=line_truth[2],dashes=line_truth[3],alpha=line_truth[4])
plt.xlabel('iterations')
plt.ylabel('misfits')
plt.savefig(caseName + '_misFits.pdf')

print("Min of misfits: ",np.argmin(misFitsValue)+1)
print("Number of iterations: ", NUM)
