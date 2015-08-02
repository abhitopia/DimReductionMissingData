import numpy as np
import math


class DimReductionMissingData:
    def __init__(self, data, reduced_dim=100):
        self.X = data.copy()
        self.reduced_dim = reduced_dim
        self.ordering = None
        self.X_mask = None
        self.X_mean = None
        self.valid_indices = []
        self.pre_process_data()
        self.center_data()

    def pre_process_data(self):
        self.X[np.isinf(self.X)] = np.nan
        self.ordering = np.arange(self.X.shape[0])
        self.X_mask = np.isnan(self.X)

    def center_data(self):
        self.X[self.X_mask] = 0
        self.X_mean = np.zeros((1, self.X.shape[1]), dtype=np.float64)
        for i in xrange(self.X.shape[1]):
            self.X_mean[0, i] = np.mean(self.X[:, i].astype(np.float64))
            self.X[:, i] = self.X[:, i].astype(np.float64) - self.X_mean[0, i]
            self.valid_indices.append(i)
        self.X = self.X[:, self.valid_indices]
        self.X_mask = self.X_mask[:, self.valid_indices]
        self.X_mean = self.X_mean[0, self.valid_indices]
        self.X[self.X_mask] = np.nan

    def get_batch(self, batch_num, batch_size):
        index_begin = batch_num * batch_size
        index_end = min(index_begin + batch_size, self.X.shape[0] - 1)
        indices = self.ordering[index_begin:index_end]
        return self.X[indices, :], self.X_mask[indices, :]

    def get_reconstruction_error(self, X, Y, B, mask):
        X_reconst = np.dot(Y, B)
        error = X - X_reconst
        error[mask] = 0.0
        error = np.mean(error ** 2)
        return error

    def get_reduced_dimensions(self, B, X, mask, verbose=False):
        X[mask] = 0  # constant c for all n
        num_samples = X.shape[0]
        Y = np.zeros((num_samples, B.shape[0]))
        for i in range(num_samples):
            if verbose:
                print 'Sample:', i + 1, 'of ', num_samples
            B_new = B.copy()
            B_new[:, mask[i, :]] = 0
            M = np.dot(B_new, np.transpose(B_new))
            c = np.transpose(np.dot(B_new, np.transpose(X[i, :])))
            Y[i, :] = np.dot(np.linalg.pinv(M), c)
        return Y

    def get_basis_matrix(self, Y, X, mask):
        num_samples, dim = X.shape
        B = np.zeros((self.reduced_dim, dim))
        X[mask] = 0
        for i in range(dim):
            yi = Y.copy()
            yi[mask[:, i], :] = 0
            m_i = np.dot(np.transpose(yi), X[:, i])  # vector with r dim
            F_i = np.dot(np.transpose(yi), yi)
            B[:, i] = np.dot(np.linalg.pinv(F_i), m_i)
        return B

    def get_derivative_of_basis_matrix(self, Y, X, mask):
        (red_dim, data_dim) = self.B.shape
        dB = np.zeros(self.B.shape)
        X[mask] = 0
        for i in range(data_dim):
            yi = Y.copy()
            yi[mask[:, i], :] = 0
            mi = np.dot(np.transpose(yi), X[:, i])  # vector with r dim
            Fi = np.dot(np.transpose(yi), yi)
            dB[:, i] = -2 * (mi - np.dot(Fi, self.B[:, i]))
        return dB

    def optimize(self, num_epochs=1, batch_size=-1, alpha=0.99):
        batch_size = self.X.shape[0] if batch_size == -1 else batch_size
        num_batches = int(math.ceil((1.0 * self.X.shape[0]) / batch_size))
        B = np.random.randn(self.reduced_dim, self.X.shape[1])
        delta_alpha = 0.05
        cur_alpha = 0
        print "{epoch}\t\t{batch}\t\t{mse_initial}\t\t\t{mse_before_B}\t\t{mse_after}".format(epoch='Epoch'.ljust(10),
                                                                                             batch='Batch'.ljust(10),
                                                                                             mse_initial='MSE Initial'.ljust(
                                                                                                 10),
                                                                                             mse_before_B='MSE Y Optimized'.ljust(
                                                                                                 10),
                                                                                             mse_after='MSE B Optimized'.ljust(
                                                                                                 10))
        for epoch in xrange(1, num_epochs + 1):
            for batch in xrange(num_batches):
                X, mask = self.get_batch(batch, batch_size)
                mse_initial = round(self.get_reconstruction_error(X, Y, B, mask) if 'Y' in locals() else np.inf, 6)
                Y = self.get_reduced_dimensions(B, X, mask)
                mse_before_B = round(self.get_reconstruction_error(X, Y, B, mask), 6)
                B_new = self.get_basis_matrix(Y, X, mask)
                cur_alpha = min(alpha, cur_alpha + delta_alpha)
                B = cur_alpha * B + (1 - cur_alpha) * (B_new - B)
                mse_after = round(self.get_reconstruction_error(X, Y, B, mask), 6)
                print "{epoch}\t\t{batch}\t\t{mse_initial}\t\t\t{mse_before_B}\t\t\t{mse_after}".format(
                    epoch=str(epoch).ljust(10),
                    batch=str(batch).ljust(10),
                    mse_initial=str(mse_initial).ljust(10),
                    mse_before_B=str(mse_before_B).ljust(10),
                    mse_after=str(mse_after).ljust(10)
                )
        return B
