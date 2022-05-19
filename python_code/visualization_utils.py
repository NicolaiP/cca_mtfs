from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.append("../")  # go to parent dir
sys.path.append("../../")  # go to parent dir
from python_code.file_utils import load_pickle_file, make_folder
from python_code.train_cca import DataLoader

# Using seaborn's style
plt.style.use('seaborn-whitegrid')

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": False,
    "font.family": "arial",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 9,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 9,
    "figure.titlesize": 9
}

plt.rcParams.update(tex_fonts)


def plot_or_save(save_fig=False, save_path='/', save_name='average'):
    """
    Function that either plots or saves the figure
    :param save_fig:
    :param save_path:
    :param save_name:
    :return:
    """

    if save_fig:
        make_folder(save_path)
        fname = save_name + '.pdf'
        # plt.tight_layout()
        # plt.tight_layout(h_pad=0.1)
        plt.savefig(os.path.normpath(os.path.join(save_path, fname)), bbox_inches='tight')
        print('Saved', fname)
        plt.close()
    else:
        plt.show()
        plt.close()


def set_fig_size(width='article', fraction=1, subplots=(1, 1), adjusted=False, adjusted2=False):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'article':
        width_pt = 430.00462
    elif width == 'report':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    if width == 'column':
        fig_width_in = 5.2
    elif width == 'full':
        fig_width_in = 7.5
    else:
        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt

    if adjusted:
        # Figure height in inches when wanting to plot freq and landmarks together
        fig_height_in = fig_width_in * (golden_ratio + golden_ratio * 0.5) * (subplots[0] / subplots[1])
    elif adjusted2:
        # Figure height in inches when wanting to plot freq, landmarks and XYZ together
        fig_height_in = fig_width_in * (golden_ratio + golden_ratio * 1) * (subplots[0] / subplots[1])
        if fig_height_in > 8.75:
            fig_height_in = 8.75
            fig_width_in = fig_height_in / ((golden_ratio + golden_ratio * 1) * (subplots[0] / subplots[1]))
    else:
        # Figure height in inches when wanting golden ratio
        fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def set_subfig_size(comp, adjusted=False, adjusted2=False, fraction=1):
    """
    Function that determines the size of the subplot
    :param comp:
    :param adjusted:
    :param adjusted2:
    :param fraction:
    :return:
    """

    if comp > 4:
        columns3 = comp % 3
        columns4 = comp % 4

        # if the two are equally low, it will prefer four columns
        all_columns = [columns4, columns3]
        if 0 == np.argmin(all_columns):
            columns = 4
        elif 1 == np.argmin(all_columns):
            columns = 3
    else:
        columns = comp
    rows = int(np.ceil(comp / columns))
    return rows, columns, set_fig_size(fraction=fraction, subplots=(rows, columns), adjusted=adjusted,
                                       adjusted2=adjusted2)


def load_normalized_face_landmarks():
    """
    Load normalized landmark points
    :return:
    """
    normalized_face_landmarks = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

    return normalized_face_landmarks


def visual_landmarks_cca_heatmap(visual_cca, ax, title='CC'):
    """ Plots the visual cca loadings as landmarks

    :param ax:
    :param visual_cca:
    :param title:
    :return:
    """
    color_map = plt.get_cmap('Reds')
    mapcolors = [color_map(int(x * color_map.N / 100)) for x in range(100)]

    # Normalize the loadings to sum to one.
    if len(visual_cca.shape) == 2:
        visual_cca = visual_cca.squeeze()
    visual_cca = abs(visual_cca)
    if len(visual_cca) == 204:
        visual_cca = np.array([visual_cca[::3], visual_cca[1::3], visual_cca[2::3]]).T
    visual_cca = visual_cca / np.sum(visual_cca)

    # Load the normalized landmarks
    landmarks = load_normalized_face_landmarks()
    landmarks -= np.mean(landmarks, axis=0)

    max_landmarks = np.max(landmarks, axis=0)
    min_landmarks = np.min(landmarks, axis=0)

    max_landmarks += 0.1
    min_landmarks -= 0.1
    landmarks[:, 1] = -landmarks[:, 1]

    # Define ellipses based on the importance in the loadings
    ells = [Ellipse(xy=landmarks[i, :], width=0.04, height=0.04, angle=0) for i in range(len(landmarks))]
    ells_center = [Ellipse(xy=landmarks[i, :], width=0.005, height=0.005, angle=0) for i in range(len(landmarks))]
    if len(visual_cca.shape) == 2:
        mean_visual_cca = np.mean(visual_cca, axis=1)
        color_sort = np.round((mean_visual_cca / max(mean_visual_cca)) * len(mapcolors)) - 1
    else:
        color_sort = np.round((visual_cca / max(visual_cca)) * len(mapcolors)) - 1

    # Plots the ellipses
    for e, color_idx in zip(ells, color_sort):
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.5)  # how transparent or pastel the color should be
        e.set_facecolor(mapcolors[int(color_idx)])

    # Plots the center of ellipses
    for e in ells_center:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(1)  # how transparent or pastel the color should be
        e.set_facecolor(mapcolors[-1])

    ax.set_xlim(min_landmarks[0], max_landmarks[0])
    ax.set_ylim(-max_landmarks[1], -min_landmarks[0])

    # With frame around plot
    ax.set(xticks=[], yticks=[])

    if title:
        ax.text(0.5, 1.05, title, horizontalalignment='center', transform=ax.transAxes)


def visual_landmarks_cca_percentage(visual_cca, ax):
    """ Plots the visual loadings on top of landmarks, to show their contribution to the CCA

    :param visual_cca:
    :param ax:
    :return:
    """

    # Normalize the loadings to sum to one.
    if len(visual_cca.shape) == 2:
        visual_cca = visual_cca.squeeze()
    visual_cca = abs(visual_cca)
    if len(visual_cca) == 204:
        visual_cca = np.array([visual_cca[::3], visual_cca[1::3], visual_cca[2::3]]).T
    visual_cca = visual_cca / np.sum(visual_cca)

    if len(visual_cca.shape) == 2:
        x, y, z = sum(visual_cca) * 100
    else:
        x, y, z = 100 / 3, 100 / 3, 100 / 3

    # To make sure that the letters fit inside the colorbar
    x_true, y_true, z_true = x, y, z
    min_val = 8
    if any([x, y, z]) < min_val:
        if x < min_val:
            x = min_val
            x_diff = x - x_true
            y -= x_diff / 2
            z -= x_diff / 2

    # Standard plot colors
    ax.bar([0], x)
    ax.bar([0], y, bottom=x)
    ax.bar([0], z, bottom=x + y)

    ax.text(0.5, x / 200 - 0.02, 'x', horizontalalignment='center', transform=ax.transAxes, color='white', size=7)
    ax.text(0.5, x / 100 + y / 200 - 0.02, 'y', horizontalalignment='center', transform=ax.transAxes,
            color='white', size=7)
    ax.text(0.5, x / 100 + y / 100 + z / 200 - 0.03, 'z', horizontalalignment='center', transform=ax.transAxes,
            color='white', size=7)
    ax.grid()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set(ylim=[0, 100])
    sec_ax = ax.secondary_yaxis('right')
    sec_ax.set_yticks(np.round([x_true, y_true + x_true, z_true + x_true + y_true], 0))


def visual_landmarks_cca_subplot(cca_loadings, save_name='subject', save_path='plots/landmark_loadings', save_fig=False,
                                 comp_titles=[]):
    """ Subplots the visual loadings on top of landmarks

    :param cca_loadings:
    :param save_name:
    :param save_path:
    :param save_fig:
    :param comp_titles:
    :return:
    """

    # Number of components
    comp = cca_loadings.shape[1]

    # Define the size of the subplot
    rows, columns, figsize = set_subfig_size(comp)
    fig, axs = plt.subplots(rows, columns, figsize=figsize)

    if not comp_titles:
        comp_titles = list(range(1, comp + 1))

    for count, (visual_cca, ax) in enumerate(zip(cca_loadings.T, axs.flat)):
        visual_landmarks_cca_heatmap(visual_cca, ax, title=f'CC {comp_titles[count]}')

    # To make sure that unused axes appear blank
    for count, ax in enumerate(axs.flat):
        if count >= comp:
            ax.axis('off')

    plot_or_save(save_fig=save_fig, save_path=save_path, save_name=save_name)

    return


def plot_pretrained_cca_landmarks_freq(audio_feature='mod_filter', visual_feature='landmarks_3d',
                                       save_fig=True, dataset='lrs3'):
    """
    Plot landmarks and MTFs for the pretrained cca
    :param audio_feature:
    :param visual_feature:
    :param save_fig:
    :param dataset:
    :return:
    """
    # Setup the data loader
    dl = DataLoader(dataset=dataset, visual_feature=visual_feature, audio_feature=audio_feature)
    frames_per_movie = int(dl.fps * 3)
    dl.load_hdf_dataset(divisible_by=frames_per_movie)

    # Setup folders
    save_kwargs = {'save_fig': save_fig, 'save_path': dl.data_path + '/results/'}
    load_path = save_kwargs['save_path']
    make_folder(save_kwargs['save_path'])

    # Load the CCA file
    cca = load_pickle_file(load_path + '/best_cca')

    # Load information about which components that are significant
    perm_dict = load_pickle_file(load_path + 'perm_dict')
    significant_comp = [ii for ii in perm_dict['significant_comp'] if perm_dict['true_corr'][ii] > 0.01]
    if not significant_comp:
        significant_comp = np.arange(5)

    # Figure titles
    comp_titles = [ii + 1 for ii in significant_comp]
    n_components = len(significant_comp)

    # Load the data both raw and transformed data
    a_train, v_train = dl.get_all()

    # Transform data
    x_scores = cca.transform_audio(a_train, use_loadings=True)

    # Reshape data such that each movie consist of one second. This allows for permutation
    frames, feature_len = np.shape(a_train)
    n_movies = int(frames / frames_per_movie)

    # Setup the plot
    rows = n_components
    columns = 3
    fig, axes = plt.subplots(rows, columns, sharex='col',
                             figsize=set_fig_size(fraction=0.6, subplots=(rows, columns), adjusted2=True),
                             gridspec_kw={'wspace': 0, 'width_ratios': [1, 0.618, 0.075]})

    # Compute FFT of the landmarks movement or audio
    # Since all the landmarks have the same frequency shape, we just plot for one.
    all_mean_spectrum = []
    trans = x_scores[:, significant_comp]
    visual_loadings = cca.y_loadings_[:, significant_comp]

    nfft = 254
    win = signal.get_window('hamming', dl.fps)  # or boxca
    # Loop through each component to plot CC landmarks and CC audio
    for comp, (ax, comp_t) in enumerate(zip(np.array(axes.flat).reshape(rows, columns), comp_titles)):
        recon_sig = np.reshape(trans[:, comp], (n_movies, frames_per_movie, -1))

        # Compute the frequency for all the videos but only one of the features, since the magnitude spectrum
        # is the same, for a given CC, when the magnitude spectrum is normalized
        spectrum = np.zeros((recon_sig.shape[0], int(nfft / 2) - 1))
        for jj in range(recon_sig.shape[0]):
            aa = signal.welch(recon_sig[jj, :], fs=dl.fps, window=win, noverlap=dl.fps / 2, nfft=nfft, axis=0,
                              scaling='density', detrend=False)
            spectrum[jj, :] = aa[1][1:-1, 0]

        freq_cca = aa[0][1:-1]

        # Scale the spectra
        tf_sem = np.std(spectrum, 0) / np.sqrt(len(spectrum))  # Plot with SEM
        mean_spectrum = np.mean(abs(spectrum), 0)
        tf_min = mean_spectrum - tf_sem
        tf_max = mean_spectrum + tf_sem
        tf_min = tf_min / max(tf_max)
        mean_spectrum = mean_spectrum / max(tf_max)
        tf_max = tf_max / max(tf_max)

        scaled_mean_spectrum = mean_spectrum
        all_mean_spectrum.append(scaled_mean_spectrum)

        # plot magnitude spectrum of CC signal
        ax[0].plot(freq_cca, scaled_mean_spectrum, c=(0.89, 0.54, 0.55))
        ax[0].plot(freq_cca, tf_max, lw=0.2, c=(0.89, 0.54, 0.55))
        ax[0].plot(freq_cca, tf_min, lw=0.2, c=(0.89, 0.54, 0.55))
        ax[0].fill_between(freq_cca, tf_min, tf_max, alpha=0.2, color=(0.89, 0.54, 0.55))
        ax[0].set(xticks=np.arange(0, dl.fps / 2, step=2), xlim=[0, dl.fps / 2],
                  yticks=np.arange(0, 1.2, step=0.2), ylim=[0, 1.005])

        plt.text(0.809, 1.04, f'CC {comp_t}', horizontalalignment='center', transform=ax[0].transAxes)

        # Plot the CC landmarks based on the loadings
        visual_landmarks_cca_heatmap(visual_loadings[:, comp], ax[1], title='')
        visual_landmarks_cca_percentage(visual_loadings[:, comp], ax[2])

        if comp == n_components - 1:
            ax[0].set(xlabel='Frequency (Hz)')

    # Trick to make the text fit on the plot
    fig.add_subplot(111, frameon=False)
    plt.grid()
    fig.text(0.99, 0.5, 'Relative importance (%)', va='center', rotation='vertical')
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.ylabel('Normalized amplitude')
    plot_or_save(save_name=f'cca_landmarks', **save_kwargs)

    return


if __name__ == '__main__':
    plot_pretrained_cca_landmarks_freq(audio_feature='mod_filter', visual_feature='landmarks_3d',
                                       save_fig=True, dataset='lrs3')
    pass
