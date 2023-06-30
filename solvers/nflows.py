from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from nflows.flows.autoregressive import MaskedAutoregressiveFlow as MAF
    from nflows.flows.realnvp import SimpleRealNVP as RealNVP

    import torch


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "nflows"

    # install_cmd = "pip"
    # requirements = ["nflows"]

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "lr": [5e-3],
        "batch_size": [64],
        "flow": ["maf", "realnvp"],
        "transforms": [1, 3, 5],
        "randperm": [True, False],
        "num_blocks_per_layer": [2],  # default from sbi
        "hidden_features": [50],  # default from sbi
    }

    def set_objective(self, X_train):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X_train = X_train

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        flow = MAF if self.flow == "maf" else RealNVP

        kwargs = {}
        if self.flow == "maf":
            kwargs["use_random_permutations"] = self.randperm

        flow = flow(
            self.X_train.shape[-1],
            num_layers=self.transforms,
            num_blocks_per_layer=self.num_blocks_per_layer,
            hidden_features=self.hidden_features,
            **kwargs,
        )

        optimizer = torch.optim.Adam(flow.parameters(), lr=self.lr)
        train_loader = torch.utils.data.DataLoader(
            self.X_train, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(n_iter + 1):
            for batch_x in train_loader:
                optimizer.zero_grad()
                loss = -flow.log_prob(batch_x).mean()
                loss.backward()
                optimizer.step()

        self.flow = flow  # needed to apply .log_prob in Objective.compute

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.flow
