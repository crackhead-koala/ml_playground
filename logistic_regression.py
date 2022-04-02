import numpy as np


class LogisticRegression:
    _cutoff: np.float64
    _learning_rate: np.float64
    _tolerance: np.float64
    _max_iter: np.int64 | float

    _coefs: np.ndarray


    def __init__(
        self,
        cutoff: float | np.float_ = 0.5,
        learning_rate: float | np.float_= 0.1,
        tolerance: float | np.float_ = 1e-5,
        max_iter: int | np.int_ | float = 10_000
    ) -> None:

        self._cutoff = np.float64(cutoff)
        self._learning_rate = np.float64(learning_rate)
        self._tolerance = np.float64(tolerance)
        self._max_iter = max_iter if np.isinf(max_iter) else np.int64(max_iter)

    def cross_entropy_loss(self,
        y_true: np.ndarray,
        y_hat: np.ndarray
    ) -> np.float_:

        return (
            -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)).sum()
        )

    def sigmoid(self, xs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-xs))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        # add column for bias
        X_ = np.c_[np.ones(X.shape[0]), X]

        # initialize coefficients at random, b ~ N(0, 1)
        self._coefs = np.random.normal(0, 1, X_.shape[1])

        # calculate initial guess loss
        y_pred = self.predict(X, True)
        loss = self.cross_entropy_loss(y, y_pred)

        # iteration counter
        n_iter = 1

        while True:

            # update learning rate
            lr = self._learning_rate * np.power(n_iter, -0.5)

            # update coefficients
            grad = X_.T @ (self.sigmoid(X_ @ self._coefs) - y)
            grad_norm = np.linalg.norm(grad)
            self._coefs -= lr * grad / grad_norm

            # calculate loss
            y_pred = self.predict(X, True)
            new_loss = self.cross_entropy_loss(y, y_pred)

            # print progress
            if n_iter % 100 == 0:
                print(
                    f'iteration {n_iter:>5} | loss: {new_loss:.4f} | ' \
                    f'learning rate: {lr:.4f}'
                )

            # check stopping conditions
            if n_iter == self._max_iter \
                or np.abs(loss - new_loss) < self._tolerance:
                print(f'done in {n_iter} iteartions | loss: {new_loss:.4f}')
                break

            loss = new_loss
            n_iter += 1

    def predict(self, X: np.ndarray, return_probs: bool = False) -> np.ndarray:

        # add column for bias
        X_ = np.c_[np.ones(X.shape[0]), X]

        likelihoods = X_ @ self._coefs
        probs = self.sigmoid(likelihoods)

        if return_probs:
            return probs

        pred_labes = \
            np.array([1 if prob >= self._cutoff else 0 for prob in probs])

        return pred_labes
