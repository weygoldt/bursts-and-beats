import matplotlib as mpl

mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.linewidth'] = 1

# lines
# mlt.rcParams['lines.linestyle'] = '--'
mpl.rcParams['lines.color'] = 'tab:blue'
mpl.rcParams['lines.linewidth'] = 1

# Axes
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# xticks
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1
# ytikcs
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1

# legend
mpl.rcParams['legend.loc'] = 'upper right'
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.fontsize'] = 16

# scatter
mpl.rcParams['scatter.marker'] = 'o'
mpl.rcParams['scatter.edgecolors'] = 'tab:orange'

# saving Figures
mpl.rcParams['figure.figsize'] = 6.4, 4.8
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.compression'] = 0