"""
An implementation of SVMs using cvxopt.

"""
import warnings
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


def kernel_dot(X1, X2, kernel_params):
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return (X1 * X2).sum(1)
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * (X1 * X2).sum(1) + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        return np.exp(-kp['gamma'] * ((X1 - X2)**2).sum(1))
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * (X1 * X2).sum(1) + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def kernel_matrix(X1, X2, kernel_params):
    kp = kernel_params
    if kp['kernel'] == 'linear':
        return X1 @ X2.T
    elif kp['kernel'] == 'poly':
        return (kp['gamma'] * X1 @ X2.T + kp['coef0']) ** kp['degree']
    elif kp['kernel'] == 'rbf':
        pw_norm = ((np.expand_dims(X1, 1) - np.expand_dims(X2, 0))**2).sum(2)
        return np.exp(-kp['gamma'] * pw_norm)
    elif kp['kernel'] == 'sigmoid':
        return np.tanh(kp['gamma'] * X1 @ X2.T + kp['coef0'])
    else:
        raise ValueError(f"Unknown parameter: {kp['kernel']}")


def get_qp_params(X, y, C, kernel_params):
    # Number of samples
    n_samples = X.shape[0]
    # Compute Gram matrix
    K = kernel_matrix(X, X, kernel_params)
    # P = y_i y_j K(x_i, x_j)
    P = np.outer(y, y) * K
    # q = -1 vector
    q = -np.ones(n_samples, dtype=np.float_)
    # Constraints: 0 <= alpha_i <= C
    # G alpha <= h
    G_std = -np.eye(n_samples)
    h_std = np.zeros(n_samples, dtype=np.float_)
    G_slack = np.eye(n_samples)
    h_slack = C * np.ones(n_samples, dtype=np.float_)
    G = np.vstack((G_std, G_slack))
    h = np.hstack((h_std, h_slack))
    # Equality constraint: sum_i y_i alpha_i = 0
    A = y.reshape(1, -1).astype(np.float_)
    b = np.array([0.0], dtype=np.float_)
    # ensure float
    P = P.astype(np.float_)
    # return
    return P, q, G, h, A, b


def fit_bias(X, y, alpha, kernel_params):
    # Support vectors where alpha > threshold
    sv = alpha > 1e-4
    if not np.any(sv):
        return 0.0
    X_sv = X[sv]
    y_sv = y[sv]
    alpha_sv = alpha[sv]
    # Compute bias for each support vector
    # b_i = y_i - sum_j alpha_j y_j K(x_j, x_i)
    K_sv = kernel_matrix(X_sv, X, kernel_params)  # shape (n_sv, n_samples)
    decision_sv = (alpha * y) @ kernel_matrix(X, X_sv, kernel_params).T
    # Actually easier: for each i in sv: sum_j alpha_j y_j K(x_j, x_i)
    temp = np.dot(alpha_sv * y_sv, K_sv)
    b_vals = y_sv - temp
    # average
    return np.mean(b_vals)


def decision_function(X, X_train, y_train, b, alpha, kernel_params):
    # Compute kernel between test points and training
    K = kernel_matrix(X, X_train, kernel_params)  # shape (n_test, n_train)
    # Weighted sum
    decision = K.dot(alpha * y_train) + b
    return decision


class CVXOPTSVC:
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma=1.0,
        coef0=0.0
    ):
        self.C = C
        self.kernel_params = {
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0
        }

    @staticmethod
    def _H_linear(X, y):
        Xy = X * np.expand_dims(y, 1)
        return Xy @ Xy.T

    def fit(self, X, y):
        # Initialize and computing H. Note the 1. to force to float type
        y = y * 2 - 1  # transform to [-1, 1]
        self.X_train = X.copy()
        self.y_train = y.copy()

        # Convert into cvxopt format
        _P, _q, _G, _h, _A, _b = get_qp_params(X, y, self.C, self.kernel_params)
        P = cvxopt_matrix(_P)
        q = cvxopt_matrix(np.expand_dims(_q, 1))
        G = cvxopt_matrix(_G)
        h = cvxopt_matrix(_h)
        A = cvxopt_matrix(_A)
        b = cvxopt_matrix(_b)

        # Run solver
        cvxopt_solvers.options['show_progress'] = False
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        self.alpha = np.array(sol['x']).squeeze(1)

        self.support_ = np.where(self.alpha > 1e-4)
        self.b = fit_bias(X, y, self.alpha, self.kernel_params)

        return self

    def decision_function(self, X):
        return decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)

    def predict(self, X):
        h = decision_function(X, self.X_train, self.y_train, self.b, self.alpha, self.kernel_params)
        return (h >= 0).astype(np.int_)

    def score(self, X, y):
        return (self.predict(X) == y).mean()
