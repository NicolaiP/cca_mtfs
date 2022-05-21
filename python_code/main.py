from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import webbrowser
import os

sys.path.append("../")  # go to parent dir
sys.path.append("../../")  # go to parent dir
from python_code.train_cca import cca_cv
from python_code.visualization_utils import plot_pretrained_cca_landmarks_freq
from python_code.file_utils import make_folder
from python_code.settings import grid_path_init, lrs3_path_init

# Parameters for training the CCA model
audio_feature = 'mod_filter'
visual_feature = 'landmarks_3d'
save_fig = True
dataset = 'lrs3'

# Make sure that the right folder structures are created automatically.
# The HDF files with the data should be stored in these folders
if dataset == 'lrs3':
    # If you run the program for the first time save the data here:
    # ../cca_mtfs/data/lrs3/lrs3.hdf5
    make_folder(lrs3_path_init())
    if not os.path.exists(lrs3_path_init() + '/lrs3.hdf5'):
        webbrowser.open('https://files.dtu.dk/u/lHb3PZJ29x-Gdvc_/e21ebbf6-f62a-4f97-a351-e34f97165172?l')
elif dataset == 'grid':
    # If you run the program for the first time save the data here:
    # ../cca_mtfs/data/grid/grid.hdf5
    make_folder(grid_path_init())
    if not os.path.exists(grid_path_init() + '/grid.hdf5'):
        webbrowser.open('https://files.dtu.dk/u/hZzuHIOc7gCXnO4e/6faa141a-effd-4c2c-92ad-75eb537de192?l')
else:
    # By default the code runs with the lrs3 datasets
    dataset = 'lrs3'
    make_folder(lrs3_path_init())
    if not os.path.exists(lrs3_path_init() + '/lrs3.hdf5'):
        webbrowser.open('https://files.dtu.dk/u/lHb3PZJ29x-Gdvc_/e21ebbf6-f62a-4f97-a351-e34f97165172?l')

# Train the CCA model
cca_cv(audio_feature=audio_feature, visual_feature=visual_feature, save_fig=save_fig, dataset=dataset)

# Plot the results
plot_pretrained_cca_landmarks_freq(audio_feature=audio_feature, visual_feature=visual_feature, save_fig=save_fig, dataset=dataset)


if __name__ == '__main__':
    print('Done!')
