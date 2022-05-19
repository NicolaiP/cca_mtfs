"""Regularized canonical correlation analysis (CCA)
Adapted from https://github.com/gallantlab/pyrcca
"""
from abc import ABC

import numpy as np
from scipy.linalg import eigh
from sklearn import preprocessing


class Model(object):

    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    def transform(self, *args, **kwargs):
        raise NotImplementedError()

    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError()

    def center_scale(self, X):
        x_scale = preprocessing.StandardScaler()
        X_scaled = x_scale.fit_transform(X)
        return X_scaled, x_scale

    def center_scale_xy(self, X, Y):
        X_scaled, x_scale = self.center_scale(X)
        Y_scaled, y_scale = self.center_scale(Y)
        return X_scaled, Y_scaled, x_scale, y_scale

    def check_is_fitted(self, attributes, msg=None, all_or_any=all):
        """Perform is_fitted validation for estimator.

        Checks if the estimator is fitted by verifying the presence of
        "all_or_any" of the passed attributes and raises a NotFittedError with the
        given message.

        Parameters
        ----------
        estimator : estimator instance.
            estimator instance for which the check is performed.

        attributes : attribute name(s) given as string or a list/tuple of strings
            Eg.:
                ``["coef_", "estimator_", ...], "coef_"``

        msg : string
            The default error message is, "This %(name)s instance is not fitted
            yet. Call 'fit' with appropriate arguments before using this method."

            For custom messages if "%(name)s" is present in the message string,
            it is substituted for the estimator name.

            Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

        all_or_any : callable, {all, any}, default all
            Specify whether all or any of the given attributes must exist.

        Returns
        -------
        None

        Raises
        ------
        NotFittedError
            If the attributes are not found.
        """
        if msg is None:
            msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this method.")

        if not hasattr(self, 'fit'):
            raise TypeError("%s is not an estimator instance." % self)

        if not hasattr(self, 'fit_transform'):
            raise TypeError("%s is not an estimator instance." % self)

        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]

        if not all_or_any([hasattr(self, attr) for attr in attributes]):
            raise NotFittedError(msg % {'name': type(self).__name__})


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """


class _CCABase(Model, ABC):
    def __init__(self, reg=None, n_components=None, cutoff=1e-15):
        self.reg = reg
        self.n_components = n_components
        self.cutoff = cutoff

    def fit(self, X, Y):
        """
        Fits the CCA model
        :param X:
        :param Y:
        :return:
        """
        # print('Training CCA, regularization = %0.4f, %d components' % (self.reg, self.n_components))

        X = X.copy()
        Y = Y.copy()

        # Subtract mean and divide by std to do this the StandardScaler is used
        X, Y, self.x_scale, self.y_scale = self.center_scale_xy(X, Y)
        data = [X, Y]

        # Get dimensions of data and number of canonical components
        kernel = [d.T for d in data]
        nDs = len(kernel)
        nFs = [k.shape[0] for k in kernel]
        numCC = min([k.shape[1] for k in kernel]) if self.n_components is None else self.n_components

        # Get the auto- and cross-covariance matrices
        crosscovs = [np.dot(ki, kj.T) / len(ki.T - 1) for ki in kernel for kj in kernel]

        # Allocate left-hand side (LH) and right-hand side (RH):
        LH = np.zeros((sum(nFs), sum(nFs)))
        RH = np.zeros((sum(nFs), sum(nFs)))

        # Fill the left and right sides of the eigenvalue problem
        # Eq. (7) in https://www.frontiersin.org/articles/10.3389/fninf.2016.00049/full
        for ii in range(nDs):
            RH[sum(nFs[:ii]):sum(nFs[:ii + 1]), sum(nFs[:ii]):sum(nFs[:ii + 1])] = \
                (crosscovs[ii * (nDs + 1)] + self.reg[ii] * np.eye(nFs[ii]))

            for jj in range(nDs):
                if ii != jj:
                    LH[sum(nFs[:jj]): sum(nFs[:jj + 1]), sum(nFs[:ii]): sum(nFs[:ii + 1])] = crosscovs[nDs * jj + ii]

        # The matrices are symmetric, i.e. A = A^T, this makes sure that small differences are evened out.
        LH = (LH + LH.T) / 2.
        RH = (RH + RH.T) / 2.

        maxCC = LH.shape[0]

        # Solve the generalized eigenvalue problem for the two symmetric matrices
        # Returns the eigenvalues and the eigenvectors
        r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
        r[np.isnan(r)] = 0

        # Sort the eigenvalues
        rindex = np.argsort(r)[::-1]
        Vs = Vs[:, rindex]

        # Extract the weights from the eigenvectors
        self.x_weights_ = Vs[:nFs[0]]
        self.y_weights_ = Vs[nFs[0]:]

        self.x_loadings_ = np.dot(self.x_weights_.T, crosscovs[0]).T
        self.y_loadings_ = np.dot(self.y_weights_.T, crosscovs[-1]).T

        return self

    def transform(self, X, Y, use_loadings=False):
        """
        This computes the scores or the components of the CCA by dotting the inputs with the learned weights.
        It has been verified that it should be np.dot(X, self.x_weights_) and not np.dot(self.x_weights_.T, X.T)
        :param X:
        :param Y:
        :return:
        """
        x_scores = self.transform_audio(X, use_loadings)
        y_scores = self.transform_video(Y, use_loadings)

        return x_scores, y_scores

    def transform_audio(self, X, use_loadings=False):
        """
        This computes the scores or the components of the CCA by dotting the inputs with the learned weights.
        It has been verified that it should be np.dot(X, self.x_weights_) and not np.dot(self.x_weights_.T, X.T)
        :param X:
        :param use_loadings:
        :return:
        """
        self.check_is_fitted('x_scale')
        X = X.copy()
        X = self.x_scale.transform(X)
        if use_loadings:
            x_scores = np.dot(X, self.x_loadings_)
        else:
            x_scores = np.dot(X, self.x_weights_)
        return x_scores

    def transform_video(self, Y, use_loadings=False):
        """
        This computes the scores or the components of the CCA by dotting the inputs with the learned weights.
        It has been verified that it should be np.dot(X, self.x_weights_) and not np.dot(self.x_weights_.T, X.T)
        :param Y:
        :param use_loadings:
        :return:
        """
        self.check_is_fitted('y_scale')
        Y = Y.copy()
        Y = self.y_scale.transform(Y)
        if use_loadings:
            y_scores = np.dot(Y, self.y_loadings_)
        else:
            y_scores = np.dot(Y, self.y_weights_)
        return y_scores


class CCA(_CCABase):
    """Attributes:
        reg (float): regularization parameter. Default is 0.1.
        n_components (int): number of canonical dimensions to keep. Default is 10.
        kernelcca (bool): kernel or non-kernel CCA. Default is True.
        ktype (string): type of kernel used if kernelcca is True.
                        Value can be 'linear' (default) or 'gaussian'.
        verbose (bool): default is True.
    Returns:
        ws (list): canonical weights
        comps (list): canonical components
        cancorrs (list): correlations of the canonical components
                         on the training dataset
        corrs (list): correlations on the validation dataset
        preds (list): predictions on the validation dataset
        ev (list): explained variance for each canonical dimension
    """

    def __init__(self, reg1=0.2, reg2=0.2, n_components=10, cutoff=1e-15):
        reg = [reg1, reg2]
        super(CCA, self).__init__(reg=reg, n_components=n_components, cutoff=cutoff)


def _listcorr(a):
    """Returns pairwise row correlations for all items in array as a list of matrices
    """
    corrs = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0, 1]) for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs


if __name__ == '__main__':

    comp = 10
    cca = CCA(n_components=comp, reg1=0.1, reg2=0.1)

    # Create some random data
    np.random.seed(1)
    audio = np.random.randn(2500, 100)
    landmarks = audio[:, :90] + (np.random.randn(2500, 90))

    # Normalize the data, as the distance heavily depends on it.
    audio = preprocessing.scale(audio)
    landmarks = preprocessing.scale(landmarks)

    cca.fit(audio, landmarks)
    a_score, v_score = cca.transform(audio, landmarks)
    true_corr = np.diag(np.corrcoef(a_score.T, v_score.T)[comp:, :comp])
    print('Done')
    print(f'This is the correlation values {true_corr}')
    pass
