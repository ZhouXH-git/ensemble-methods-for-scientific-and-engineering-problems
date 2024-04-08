#!/usr/bin/env python3
""" Plotting script for the periodic hill OpenFOAM case. """

# standard library imports
import os
from matplotlib import markers
from matplotlib.pyplot import MultipleLocator

# third party imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# parameters settings
nsamples = 20
niter = 8

# for x cross sections
C_y = 103
stp_y=np.array([3, 8])
# for centerline
C_x = 119
stp_x=np.array([7, 5])
NU_WALL=1.5598e-05                  # the kinetic viscosity at the wall
D_jet = 25.4/1000*2                 # diameter of the jet exit

# set plot properties
params = {
    'text.latex.preamble': '\\usepackage{gensymb}',
    'text.usetex': True,
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'magma',
    'axes.grid': False,
    'savefig.dpi': 600,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'font.size': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': [15, 12.5],
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'lines.linewidth': 2,
    'lines.markersize': 6,
}
mpl.rcParams.update(params)


def plot_scalar_truth(filename, ax, index1, index2, k, axis_aspect, ls, cl, lw, dash, alpha):
    # index1: for velocity u - 2 or v - 3
    # index2: for coordinate x - 0 or y - 1

    data = np.loadtxt(filename, comments='%')

    if k == 5:
        xval = data[:,index1]
        yval = data[:,index2-1]
        plot, = ax.plot(yval, xval, linestyle=ls, color=cl, lw=lw, dashes=dash, alpha=alpha)
        ax.set_aspect(axis_aspect[1])
    else:
        xval = data[:,index1]
        yval = data[:,index2]
        plot, = ax.plot(xval, yval, linestyle=ls, color=cl, lw=lw, dashes=dash, alpha=alpha)
        ax.set_aspect(axis_aspect[0])
    return plot, xval, yval


def plot_scalar_obs(filename, ax, index1, index2, k, field, axis_aspect, C_x, stp_x, C_y, stp_y, ls, cl, mk, ms, mew):
    # index1: for velocity u - 2 or v - 3
    # index2: for coordinate x - 0 or y - 1

    NASA_SYM = 69

    if field == 'Ux':
        data = np.loadtxt(filename, comments='%')    
        if k == 5:
            # the plot will only show 12 points in the figure of centerline: the axis limit
            xval1 = data[0:C_x:stp_x[0],index1]
            xval2 = data[C_x::stp_x[1],index1]
            xval = np.hstack([xval1, xval2])
            yval1 = data[0:C_x:stp_x[0],index2-1]
            yval2 = data[C_x::stp_x[1],index2-1]
            yval = np.hstack([yval1, yval2])

            plot, = ax.plot(yval, xval, linestyle=ls, color=cl, marker=mk, markersize=ms, markeredgewidth=mew)
            ax.set_aspect(axis_aspect[1])

            return plot, xval, yval



def plot_scalar_of(filename, ax, index, k, axis_aspect, U_jet, ls, cl, lw, dash, alpha):
    # index1: for velocity u - 1 or v - 2

    data = np.loadtxt(filename, comments='%')
    xval=data[:,index]/U_jet
    yval=data[:,0]/D_jet
    if k == 5:
        plot, = ax.plot(yval, xval, linestyle=ls, color=cl, lw=lw, dashes=dash, alpha=alpha)
        ax.set_aspect(axis_aspect[1])
    else:
        plot, = ax.plot(xval, yval, linestyle=ls, color=cl, lw=lw, dashes=dash, alpha=alpha)
        ax.set_aspect(axis_aspect[0])
    return plot, xval, yval


def main():
    # options
    savefig = True
    showfig = False
    legend_top = True
    x_pos_list = [2, 5, 10, 15, 20]

    x_name_list=[]
    for i in range(5):
        x_name_list.append('x_'+str(x_pos_list[i]))
    x_name_list.append('center')

    # define color for different profiles
    # [line_style, color, line_width, dashes, opacity]
    line_truth = ['-', 'k', 2, (None, None), 1.0]
    line_base = ['-', 'tab:red', 2, (9, 3), 1.0]

    line_inferred = ['-', 'tab:blue', 2, (5, 2), 1.0]
    line_samples = ['-', 'tab:grey', 0.1, (None, None), 0.5]

    # line_style, color, marker, markersize, markeredgewidth
    line_obs = ['','black', 'x', 6, 1.5]

    # legend names
    truth_name = 'truth'
    base_name = 'k-omega'
    inferred_name = 'learned model'
    samples_name = 'Samples'
    obs_name = "Observations"

    # subfigs titles
    subfigs_names = ['x = 2', 'x = 5', 'x = 10', 'x = 15', 'x = 20', 'jet centerline']

    # create figure
    def plot_figure(case, field):
        if case == 'prior':
            iter = 0
        elif case == 'posterior':
            iter = niter
        else:
            raise ValueError("'case' must be one of 'prior' or 'posterior'")

        # import pdb; pdb.set_trace()
        if field == 'Ux':
            field_filename = 'U'
            index1 = 1
            index2 = 2
            index3 = 1
            axis_plot = [[0, 1.2, 0, 1.5], [0, 22, 0, 1.1]]
            axis_locator = np.array([[0.2, 0.5, 0.05, 0.1],[5, 0.2, 1, 0.05]])
            axis_aspect = [1, 25]
        elif field == 'Uy':
            field_filename = 'U'
            index1 = 3
            index2 = 3
            index3 = 1
            axis_plot = [[-0.04, 0.06, 0, 1.5], [0, 22, -0.001, 0.001]]
            axis_locator = np.array([[0.02, 0.5, 0.005, 0.1],[5, 0.0005, 1, 0.0001]])
            axis_aspect = [0.088, 14400]
        else:
            raise ValueError("'field_name' must be 'Ux' or 'Uy'")

        # start figure and plot domain
        fig, axes = plt.subplots(2, 3)
              
        axes = axes.flatten()

        for k in range(6):
            
            axk = axes[k]

            # plot profiles
            plot_of = plot_scalar_of

            # samples
            xname = x_name_list[k]        
            ls = line_samples
            sample_mean = 0
            for i in range(nsamples):
                # get the U_jet
                filename_U_jet = f"results_ensemble/sample_{i:d}/foam_base_ASJet/postProcessing/" +\
                    f"sampleDict/{iter}/line_jet_exit_{field_filename}.xy"
                u_jet_exit=np.loadtxt(filename_U_jet, comments='%')
                U_jet=u_jet_exit[0,1]

                # get the target cross section
                filename = f"results_ensemble/sample_{i:d}/foam_base_ASJet/postProcessing/" +\
                    f"sampleDict/{iter}/line_{xname}_{field_filename}.xy"
                norm = 1
                # index1=1                     # velocity u
                _, xval, yval = plot_of(filename, axk, index1, k, axis_aspect, U_jet, ls[0], ls[1], ls[2], ls[3], ls[4])
                
                sample_mean += xval     # update mean: the variable is xval

            sample_mean /= nsamples
            ls = line_inferred
            if k == 5:
                _ = axk.plot(yval, sample_mean, linestyle=ls[0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4])
                ASJ_center_mean = np.vstack((yval, sample_mean)).T
                np.savetxt("ASJ_center_mean_kOmegaQuadratic.dat", ASJ_center_mean)
            elif k == 2:
                _ = axk.plot(sample_mean, yval, linestyle=ls[0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4])
                ASJ_10_mean = np.vstack((sample_mean, yval)).T
                np.savetxt("ASJ_10_mean_kOmegaQuadratic.dat", ASJ_10_mean)
            else:
                _ = axk.plot(sample_mean, yval, linestyle=ls[0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4])

            # truth
            ls = line_truth
            filename = 'inputs/ASJet_simpleFoam_kOmega/postProcessing/sampleDict/'
            filename += f'0/NASA_{xname}.dat'
            _ = plot_scalar_truth(filename, axk, index2, index3, k, axis_aspect, ls[0], ls[1], ls[2], ls[3], ls[4])

            # observations
            ls = line_obs
            filename = 'inputs/ASJet_simpleFoam_kOmega/postProcessing/sampleDict/'
            filename += f'0/NASA_{xname}.dat'

            _ = plot_scalar_obs(filename, axk, index2, index3, k, field, axis_aspect, C_x, stp_x, C_y, stp_y, ls[0], ls[1], ls[2], ls[3], ls[4])
            
            # baseline: k - omega
            ls = line_base
            # get the baseline's U_jet
            filename_U_jet = 'inputs/ASJet_simpleFoam_kOmega/postProcessing/sampleDict/'
            filename_U_jet += f'60000/line_jet_exit_{field_filename}.xy'
            u_jet_exit=np.loadtxt(filename_U_jet, comments='%')
            U_jet=u_jet_exit[0,1]

            filename = 'inputs/ASJet_simpleFoam_kOmega/postProcessing/sampleDict/'
            filename += f'60000/line_{xname}_{field_filename}.xy'
            # index1=1                     # velocity u
            _ = plot_of(filename, axk, index1, k, axis_aspect, U_jet, ls[0], ls[1], ls[2], ls[3], ls[4])

            # set figure properties and labels

            if k == 5:
                axk.axis(axis_plot[1])
                if field == 'Ux':
                    axk.set_ylabel(r'$u/U_{jet}$')
                else:
                    axk.set_ylabel(r'$v/U_{jet}$')
                axk.set_xlabel(r'$x/D_{jet}$')
                x_major_locator=MultipleLocator(axis_locator[1,0])
                y_major_locator=MultipleLocator(axis_locator[1,1])
                axk.xaxis.set_major_locator(x_major_locator)
                axk.yaxis.set_major_locator(y_major_locator)
                xminorLocator = MultipleLocator(axis_locator[1,2])
                yminorLocator = MultipleLocator(axis_locator[1,3])
                axk.xaxis.set_minor_locator(xminorLocator)
                axk.yaxis.set_minor_locator(yminorLocator)
            else:
                axk.axis(axis_plot[0])
                if field == 'Ux':
                    axk.set_xlabel(r'$u/U_{jet}$')
                else:
                    axk.set_xlabel(r'$v/U_{jet}$')
                axk.set_ylabel(r'$y/D_{jet}$')
                x_major_locator=MultipleLocator(axis_locator[0,0])
                y_major_locator=MultipleLocator(axis_locator[0,1])
                axk.xaxis.set_major_locator(x_major_locator)
                axk.yaxis.set_major_locator(y_major_locator)
                xminorLocator = MultipleLocator(axis_locator[0,2])
                yminorLocator = MultipleLocator(axis_locator[0,3])
                axk.xaxis.set_minor_locator(xminorLocator)
                axk.yaxis.set_minor_locator(yminorLocator)

            axk.set_title(subfigs_names[k])
        
        # legend: use the same line type of corresponding data
        lines = []
        labels = []

        ls = line_truth
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(truth_name)

        ls = line_obs
        lines.append(Line2D(
            [0], [0], linestyle='', color=ls[1], marker=ls[2]))
        labels.append(obs_name)

        ls = line_base
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(base_name)

        ls = line_samples
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(samples_name)

        ls = line_inferred
        lines.append(Line2D(
            [0], [0], color=ls[1], lw=ls[2], dashes=ls[3], alpha=ls[4]))
        labels.append(inferred_name)

        # plot the legend: on the whole figure's top
        if legend_top:
            fig.legend(lines, labels, handlelength=4,
                    loc='center', bbox_to_anchor=(0.5, 0.975),
                    fancybox=False, shadow=False, ncol=5)                    
        else:
            fig.legend(lines, labels, handlelength=4,
                    loc='center left', bbox_to_anchor=(1.0, 0.5),
                    fancybox=False, shadow=False)        

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)

        # save/show
        if savefig:
            figname = f"ASJet_{field}_{case}"
            plt.savefig(f"{figname}.pdf")

    plot_figure('prior', 'Ux')
    plot_figure('posterior', 'Ux')
    

    if showfig:
        plt.show()


if __name__ == "__main__":
    main()
