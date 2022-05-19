from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../")  # go to parent dir
sys.path.append("../../")  # go to parent dir
from python_code.train_cca import cca_cv
from python_code.visualization_utils import plot_pretrained_cca_landmarks_freq
from python_code.file_utils import make_folder
from python_code.settings import grid_path_init, lrs3_path_init

# Make sure that the folders are created.
# The HDF files with the data should be stored in these folders
make_folder(lrs3_path_init())
make_folder(grid_path_init())

# Parameters for training the CCA model
audio_feature = 'mod_filter'
visual_feature = 'landmarks_3d'
save_fig = True
dataset = 'lrs3'

cca_cv(audio_feature=audio_feature, visual_feature=visual_feature, save_fig=save_fig, dataset=dataset)
plot_pretrained_cca_landmarks_freq(audio_feature=audio_feature, visual_feature=visual_feature, save_fig=save_fig, dataset=dataset)


if __name__ == '__main__':
    pass
