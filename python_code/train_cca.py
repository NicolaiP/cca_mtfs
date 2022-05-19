from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import h5py
import numpy as np
import os
import time
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

sys.path.append("../")  # go to parent dir
sys.path.append("../../")  # go to parent dir
from python_code.rcca import CCA
from python_code import settings
from python_code.file_utils import make_folder, save_as_pickle


class DataLoader(object):
    """
    The DataLoader class loads and normalizes the data.
    """

    def __init__(self, dataset='lrs3', visual_feature='landmarks_3d_8hz', audio_feature='mod_filter'):

        self.dataset = dataset
        self.visual_feature = visual_feature
        self.audio_feature = audio_feature

        valid_dataset = ('actors', 'grid', 'lrs3', 'lrs3_test')

        if dataset not in valid_dataset:
            raise ValueError('Unknown dataset, should be one of: {}'.format(valid_dataset))

        if dataset == 'grid':
            self.data_path = settings.grid_path_init()
            self.fps = 25
        elif dataset == 'lrs3':
            self.data_path = settings.lrs3_path_init()
            self.fps = 25
        elif dataset == 'lrs3_test':
            self.data_path = settings.lrs3_test_path_init()
            self.fps = 25

        self.data_dict = {}

    def load_hdf_dataset(self, divisible_by=1, videos_2_load=None):
        """
        Loads in either actors, lrs3, or grid dataset in a dictionary.
        The divisible_by param ensures that the number of frames in a videos is divisible by divisible_by
        :param divisible_by: enables shuffling of video segments at a later stage in a safe manner.
        :param videos_2_load: If only a subset of videos needs to be loaded
        :return:
        """

        if self.dataset in ['grid', 'lrs3', 'lrs3_test']:
            hdf_file = h5py.File(os.path.join(self.data_path, self.dataset + '.hdf5'), 'r')

            # Create dictionary for each subject with a visual feature and audio feature
            self.data_dict = {subject: {self.visual_feature: list(), self.audio_feature: list()}
                              for subject in hdf_file.keys()
                              if self.visual_feature in hdf_file[subject] and self.audio_feature in hdf_file[subject]
                              and hdf_file[subject].attrs['n_frames'] >= divisible_by}

            for movie in self.data_dict.keys():
                try:
                    # Calculate the number of frames to use in the video based on the divisible by.
                    # This enables us to shuffle segments of the movie around in a safe manner.
                    frames_in_video = hdf_file[movie].attrs['n_frames']
                    frames_2_use = frames_in_video - (frames_in_video % divisible_by)
                    self.data_dict[movie][self.audio_feature] = hdf_file[movie][self.audio_feature][()][:frames_2_use]
                    self.data_dict[movie][self.visual_feature] = self.visual_norm(
                        hdf_file[movie][self.visual_feature][()][:frames_2_use])
                except:
                    continue
        else:
            raise ValueError('Unknown dataset!')

        if videos_2_load is not None:
            # If you only want to load a smaller dataset
            smaller_size = {}
            for count, movie in enumerate(self.data_dict.keys()):
                if count < videos_2_load:
                    smaller_size[movie] = self.data_dict[movie]
                else:
                    continue
            self.data_dict = smaller_size
        return

    def get_all(self):
        """
        Gets all the features. See load_hdf_dataset for how data is loaded.
        :return:
        """
        v_train = []
        a_train = []
        for train_key in self.data_dict.keys():
            v_train.extend(self.data_dict[train_key][self.visual_feature])
            a_train.extend(self.data_dict[train_key][self.audio_feature])
        a_train = np.array(a_train)
        v_train = np.array(v_train)
        return a_train, v_train

    def leave_one_out_test_train_val(self, test_size=0.1, k_fold=5):
        """
        Generator, that yields a training set, test set and a validation set
        :param test_size: Percentage of data to extract for the test set
        :param k_fold: The number of folds to use in the cross validation loop
        :return:
        """
        # Get the unique subject id for LRS3 and GRID
        if self.dataset == 'lrs3':
            s_id = 5
        elif self.dataset == 'grid':
            s_id = 6
        else:
            s_id = 5

        # Both GRID and LRS3 contain multiple videos for each subject, hence we first extract the subject id
        speaker_id = sorted(list(set([key[:-s_id] for key in self.data_dict.keys()])))

        # Randomly extract the test ids the remaining ids will be used in the training
        # The test ID's is always s20_, s2_, and s33_
        np.random.seed(0)
        test_id = list(np.random.choice(speaker_id, int(len(speaker_id) * test_size), replace=False))

        # This is a safe way of extracting the train ids in each fold, since it allows len(train_id) % k_fold != 0
        all_train_id = [speaker for speaker in speaker_id if speaker not in test_id]
        all_train_id = [all_train_id[ii::k_fold] for ii in range(k_fold)]

        a_test = np.vstack([item[self.audio_feature] for key, item in self.data_dict.items() if key[:-s_id] in test_id])
        v_test = np.vstack([item[self.visual_feature] for key, item in self.data_dict.items() if key[:-s_id] in test_id])
        for ii in range(k_fold):
            a_val = []
            v_val = []
            a_train = []
            v_train = []
            val_id = all_train_id[ii]
            for key, item in self.data_dict.items():
                # Faster to test whether the key is in the smaller test and val set than in the larger training set
                if key[:-s_id] in val_id:
                    a_val.extend(item[self.audio_feature])
                    v_val.extend(item[self.visual_feature])
                elif key[:-s_id] in test_id:
                    continue
                else:
                    a_train.extend(item[self.audio_feature])
                    v_train.extend(item[self.visual_feature])

            a_val = np.array(a_val)
            v_val = np.array(v_val)
            a_train = np.array(a_train)
            v_train = np.array(v_train)
            print(f'Run {ii + 1} out of {k_fold}, time = {time.strftime("%d/%m/%y %H:%M:%S", time.localtime())}')

            yield a_train, v_train, a_test, v_test, a_val, v_val

    @staticmethod
    def visual_norm(data):
        """
        Normalizes visual 3d landmarks across the videos, frames. Beside that it can calculate motion vector between
        adjacent frames and the distance from the point on the nose to all other points.
        :param data: visual 3d landmarks
        :return:
        """
        data_shape = data.shape
        data = data.reshape((-1, 3))
        mean_dir = np.mean(data, axis=0)
        std_dir = np.std(data, axis=0)
        data -= mean_dir
        data = data / std_dir
        data = data.reshape((data_shape[0], -1))

        return np.array(data, dtype=np.float32)


def cca_perm(X_train, Y_train, X_test, Y_test, num_perm=1000, divisible_by=25):
    """
    The objective is to maximize the total correlation between components that are significant.
    It optimizes the regularization parameter using bayesian optimization.

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_perm: Number of permutations
    :param divisible_by: The number of
    :return:
    """
    x_space = [10 ** -5, 10 ** 0]
    y_space = [10 ** -5, 10 ** 0]
    space = [Real(x_space[0], x_space[1], "log-uniform", name='reg1'),
             Real(y_space[0], y_space[1], "log-uniform", name='reg2')]
    comp = min((X_test.shape[-1], Y_test.shape[-1]))

    perm_dict = {'significant_comp': [], 'min_func': [], 'perm_corr': [], 'true_corr': []}
    gp_dict = {'x_iters': [], 'func_vals': []}
    objective_count = []

    @use_named_args(space)
    def objective(**params):

        # Extract regularization
        audio_reg = params['reg1']
        visual_reg = params['reg2']
        save_cca = True
        if [audio_reg, visual_reg] in gp_dict['x_iters']:
            # Checks whether the parameters have been evaluated before, if so it uses these values.
            loss_function = [gp_dict['func_vals'][count] for count, ii in enumerate(gp_dict['x_iters']) if
                             ii == [audio_reg, visual_reg]][0]
            save_cca = False

        else:
            cca = CCA(reg1=audio_reg, reg2=visual_reg, n_components=comp)
            cca.fit(X_train, Y_train)
            x_scores, y_scores = cca.transform(X_test, Y_test)
            true_corr = np.diag(np.corrcoef(x_scores.T, y_scores.T)[comp:, :comp])

            a_test_rand1 = X_test.copy()
            frames = len(X_test)
            a_test_rand1 = np.reshape(a_test_rand1, (-1, divisible_by, X_test.shape[-1]))
            n_movies = a_test_rand1.shape[0]

            # Run CCA on permutated data
            all_corr = np.zeros((num_perm, comp))
            np.random.seed(0)
            for ii in range(num_perm):
                # THE PERMUTATION AND RESHAPING HAS BEEN DOUBLED CHECKED AND IT WORKS LIKE A CHARM
                rand = np.random.permutation(n_movies)
                a_test_rand = a_test_rand1[rand, :, :]
                a_test_rand = np.reshape(a_test_rand, (frames, -1))
                x_scores, y_scores = cca.transform(a_test_rand, Y_test)
                all_corr[ii, :] = np.diag(np.corrcoef(x_scores.T, y_scores.T)[comp:, :comp])

            # Identify the permutations below the significance level
            perm = 1 - sum(true_corr - all_corr > 0) / num_perm
            perm_good = np.where(perm < 0.05)[0]

            # Get the median of the random permutations
            median_corr = np.median(all_corr, axis=0)

            # As the function seeks to minimizes we convert to minus area
            loss_function = -sum(true_corr[perm_good] - median_corr[perm_good])

        if not len(objective_count) < len(gp_dict['x_iters']):
            gp_dict['x_iters'].append([audio_reg, visual_reg])
            gp_dict['func_vals'].append(loss_function)

        if loss_function <= min(gp_dict['func_vals']) and save_cca:
            perm_dict['significant_comp'] = perm_good
            perm_dict['min_func'] = loss_function

        objective_count.append(1)
        return loss_function

    n_calls = 10
    n_random_starts = 5
    # Description of bayesian optimization https://distill.pub/2020/bayesian-optimization/
    function_gp = gp_minimize(objective, space, n_calls=n_calls, random_state=0, n_jobs=1, verbose=False,
                              n_random_starts=n_random_starts)

    return function_gp, perm_dict['significant_comp']


def cca_cv(audio_feature='mod_filter', visual_feature='landmarks_3d', save_fig=True, dataset='lrs3', videos_2_load=None):
    """
    Cross-validation to train the CCA.

    :param audio_feature:
    :param visual_feature:
    :param save_fig:
    :param dataset:
    :return:
    """

    dl = DataLoader(dataset=dataset, visual_feature=visual_feature, audio_feature=audio_feature)

    save_kwargs = {'save_fig': save_fig, 'save_path': dl.data_path + '/results/'}
    if save_fig:
        make_folder(save_kwargs['save_path'])

    path_joiner = lambda x_str: save_kwargs['save_path'] + x_str

    print('Results saved to:')
    print(save_kwargs['save_path'])

    divisible_by = dl.fps
    dl.load_hdf_dataset(divisible_by=divisible_by, videos_2_load=videos_2_load)

    k_folds = 5
    num_perm = 1000
    gp_dict = {'x_iters': [], 'func_vals': [], 'set': [], 'significant_comp': []}

    print('\n##############################')
    print(f'Code started {time.strftime("%d/%m/%y %H:%M:%S", time.localtime())}')
    print('##############################\n')

    # Run the cross-validation to identify the best regularization parameters
    for a_train, v_train, a_test, v_test, a_val, v_val in dl.leave_one_out_test_train_val(test_size=0.1, k_fold=k_folds):
        optimum, sig_comp = cca_perm(a_train, v_train, a_val, v_val, divisible_by=divisible_by, num_perm=num_perm)
        gp_dict['x_iters'].append(optimum['x'])
        gp_dict['func_vals'].append(optimum['fun'])
        gp_dict['set'].append('val')
        gp_dict['significant_comp'].append(sig_comp)
        save_as_pickle(gp_dict, path_joiner('gp_dict'))
        df = pd.DataFrame(gp_dict)
        df.to_csv(path_joiner('gp_dict') + '.csv', index=False)

    comp = min((a_test.shape[-1], v_test.shape[-1]))
    audio_reg, visual_reg = gp_dict['x_iters'][np.argmin(gp_dict['func_vals'])]

    # Combine the training and validation set to one big training set, to train one last model.
    # This can is to identify the significant components
    a_train_all = np.vstack([a_train, a_val])
    v_train_all = np.vstack([v_train, v_val])

    # Train the CCA model based on the best regularization parameters
    cca = CCA(reg1=audio_reg, reg2=visual_reg, n_components=comp)
    cca.fit(a_train_all, v_train_all)
    x_scores, y_scores = cca.transform(a_test, v_test)
    true_corr = np.diag(np.corrcoef(x_scores.T, y_scores.T)[comp:, :comp])

    a_test_rand1 = a_test.copy()
    frames = len(a_test)
    a_test_rand1 = np.reshape(a_test_rand1, (-1, divisible_by, a_test.shape[-1]))
    n_movies = a_test_rand1.shape[0]

    # Run CCA on permutated data to identify significant components
    perm_corr = np.zeros((num_perm, comp))
    np.random.seed(0)
    for ii in range(num_perm):
        # THE PERMUTATION AND RESHAPING HAS BEEN DOUBLED CHECKED AND IT WORKS LIKE A CHARM
        rand = np.random.permutation(n_movies)
        a_test_rand = a_test_rand1[rand, :, :]
        a_test_rand = np.reshape(a_test_rand, (frames, -1))
        x_scores, y_scores = cca.transform(a_test_rand, v_test)
        perm_corr[ii, :] = np.diag(np.corrcoef(x_scores.T, y_scores.T)[comp:, :comp])

    # Identify the permutations below the significance level
    perm = 1 - sum(true_corr - perm_corr > 0) / num_perm
    perm_good = np.where(perm < 0.05)[0]

    # Get the median of the random permutations
    median_corr = np.median(perm_corr, axis=0)

    # As the function seeks to minimizes we convert to minus area
    loss_function = -sum(true_corr[perm_good] - median_corr[perm_good])

    # Save the best model and parameters
    save_as_pickle(cca, path_joiner('best_cca'))
    gp_dict['x_iters'].append([audio_reg, visual_reg])
    gp_dict['func_vals'].append(loss_function)
    gp_dict['set'].append('test')
    gp_dict['significant_comp'].append(perm_good)
    save_as_pickle(gp_dict, path_joiner('gp_dict'))
    df = pd.DataFrame(gp_dict)
    df.to_csv(path_joiner('gp_dict') + '.csv', index=False)

    perm_dict = {'significant_comp': perm_good,
                 'min_func': loss_function,
                 'perm_corr': perm_corr,
                 'true_corr': true_corr}

    save_as_pickle(perm_dict, path_joiner('perm_dict'))


if __name__ == '__main__':
    cca_cv(audio_feature='mod_filter', visual_feature='landmarks_3d', save_fig=True, dataset='lrs3')
    pass
