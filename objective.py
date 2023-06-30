from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


def negative_loglikelihood(log_prob_fn, X_test):
    """Computes the negative log likelihood over a test set X_test.

    Parameters
    ----------
        log_prob_fn : callable
            A function that takes a test sample and returns the log
            probability of the sample under the model.


        X_test : torch.FloatTensor, shape (n_samples*test_size, n_features)
            Test set.

    Returns
    -------
        nll : float
              The negative log likelihood.
    """
    return -log_prob_fn(X_test).mean().item()


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "Negative Log Likelihood"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.

    # === not used in this example ===
    parameters = {}
    # =================================

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3"

    def set_data(self, X_train, X_test):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X_train = X_train
        self.X_test = X_test

    def compute(self, flow):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        log_prob_fn = flow.log_prob

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=negative_loglikelihood(log_prob_fn, self.X_test),
        )

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return np.mean([0] * len(self.X_test))

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(X_train=self.X_train)
