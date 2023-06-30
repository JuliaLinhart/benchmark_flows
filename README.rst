
My Benchopt Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of **Normalizing Flows**:
Conditional density esrimators trained via Maximum Likelihood Estimation (MLE).
They minimize the negative log-likelihood of the data under the model:


$$\\min_{w} - \\frac{1}{n} \\sum_{i=1}^n \\log f_w(X_i)$$


where $n$ stands for the number of data samples and $w$ are the model parameters. 


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/JuliaLinhart/benchmark_flows
   $ pip install scikit-learn
   $ pip install zuko
   $ benchopt install benchmark_flows -s zuko -d simulated
   $ benchopt run benchmark_flows 

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_flows -s zuko -d simulated --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/JuliaLinhart/benchmark_flows/workflows/Tests/badge.svg
   :target: https://github.com/JuliaLinhart/benchmark_flows/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
