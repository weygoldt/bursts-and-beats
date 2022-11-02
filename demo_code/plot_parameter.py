import matplotlib as mpl

<<<<<<< HEAD
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
=======
mpl.rcParams["axes.labelsize"] = 13
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams["ytick.labelsize"] = 11
>>>>>>> Alex
mpl.rcParams['axes.linewidth'] = 1

# lines
# mlt.rcParams['lines.linestyle'] = '--'
<<<<<<< HEAD
mpl.rcParams['lines.color'] = 'tab:blue'
mpl.rcParams['lines.linewidth'] = 1

=======
mpl.rcParams['lines.color'] = 'black'
mpl.rcParams['lines.linewidth'] = 1

# Boxplot Parameter
mpl.rcParams['boxplot.patchartist'] = True
mpl.rcParams['boxplot.showfliers'] = False
mpl.rcParams['boxplot.medianprops.color'] = 'black'
mpl.rcParams['boxplot.medianprops.linewidth'] = 1

# Text
mpl.rcParams['font.size'] = 11

>>>>>>> Alex
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
<<<<<<< HEAD
mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.fontsize'] = 16

# scatter
mpl.rcParams['scatter.marker'] = 'o'
mpl.rcParams['scatter.edgecolors'] = 'tab:orange'

# saving Figures
mpl.rcParams['figure.figsize'] = 6.4, 4.8
=======
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['legend.markerscale'] = 0.5

# scatter
mpl.rcParams['scatter.marker'] = 'o'
mpl.rcParams['scatter.edgecolors'] = 'k'


# saving Figures
mpl.rcParams['figure.figsize'] = 7.4, 6.8
>>>>>>> Alex
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['pdf.compression'] = 0