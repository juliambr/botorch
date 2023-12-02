#!/usr/bin/env python3
# Copyright (c) XXXXXXXXXXXXXXXXXXXXXXX
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Objective Modules to be used with acquisition functions."""

from __future__ import annotations

from argparse import Namespace
import copy
import inspect
import warnings
from abc import ABC, abstractmethod

import numpy as np
import torch

from botorch.acquisition.utils import compute_data_grid


class Algorithm(ABC):
    r"""Abstract base class for BAX algorithm.
    """

    def __init__(self, params: None) -> None:
        r"""Constructor for the Algorithm base class.

        Args:
            params: Parameters of the algorithm.
        """
        super().__init__()
        self.set_params(params)

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        params = Namespace(**params)

        # Set self.params
        self.params = Namespace()
        self.params.name = getattr(params, "name", "Algorithm")

    def initialize(self):
        """Initialize algorithm, reset execution path."""
        self.exe_path = Namespace(x=[], y=[])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        # Default behavior: return a uniform random value 10 times
        next_x = np.random.uniform() if len(self.exe_path.x) < 10 else None
        return next_x

    def take_step(self, f):
        """Take one step of the algorithm."""
        x = self.get_next_x()
        if x is not None:
            y = f(x)
            self.exe_path.x.append(x)
            self.exe_path.y.append(y)

        return x

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        self.initialize()

        # Step through algorithm
        x = self.take_step(f)
        while x is not None:
            x = self.take_step(f)

        # Return execution path and output
        return self.exe_path, self.get_output()

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        # As default, return untouched execution path
        return self.exe_path

    def get_copy(self):
        """Return a copy of this algorithm."""
        return copy.deepcopy(self)

    @abstractmethod
    def get_output(self):
        """Return output based on self.exe_path."""
        pass


class FixedPathAlgorithm(Algorithm):
    """
    Algorithm with a fixed execution path input sequence, specified by x_path parameter.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = Namespace(**params)

        self.params.name = getattr(params, "name", "FixedPathAlgorithm")
        self.params.x_path = getattr(params, "x_path", [])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        x_path = self.params.x_path
        next_x = x_path[len_path] if len_path < len(x_path) else None
        return next_x

    def get_output(self):
        """Return output based on self.exe_path."""
        # Default behavior: return execution path
        return self.exe_path

class PDPAlgorithm(FixedPathAlgorithm):
    """
    Algorithm that computes the PDP.
    """
    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = Namespace(**params)

        self.params.name = getattr(params, "name", "PDPAlgorithm")
        self.params.grid_size = getattr(params, "grid_size", 10)
        self.params.n_points = getattr(params, "n_points")
        self.params.xs = getattr(params, "xs")
        self.params.bounds = getattr(params, "bounds")

        self.params.x_path = compute_data_grid(self.params.xs, self.params.bounds, self.params.n_points, self.params.grid_size)

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        x_path = self.params.x_path
        next_x = x_path[len_path] if len_path < len(x_path) else None
        return next_x

    def run_algorithm_on_f(self, f):
        """
        Run the algorithm by sequentially querying function f. Return the execution path
        and output.
        """
        self.initialize()

        exe_path = Namespace()

        x_path = self.params.x_path
        x_path_f = f(x_path)
        x_path_len = x_path.size(0)
        exe_path.x = [x_path[i] for i in range(x_path_len)]
        exe_path.y = [x_path_f[i] for i in range(x_path_len)]

        self.exe_path = exe_path

        # Return execution path and output
        return self.exe_path, self.get_output()

    def get_output(self):
        """Return output based on self.exe_path."""
        # Evaluated values at the grid 
        y = torch.stack(self.exe_path.y)
        # Group along grid points 
        keys, indices = torch.unique(self.params.x_path[:,self.params.xs], return_inverse=True)
        # Compute mean as main output and variance in case we want to display uncertainty bands 
        means = torch.zeros_like(keys, dtype=self.params.x_path.dtype)
        var = torch.zeros_like(keys, dtype=self.params.x_path.dtype)
        for i, key in enumerate(keys):
            means[i] = y[indices == i].mean()
            var[i] = y[indices == i].var()
        output = Namespace(x=keys, y=means, var=var)
        return output
    



class AlgorithmSet:
    """Wrapper that duplicates and manages a set of Algorithms."""

    def __init__(self, algo):
        """Set self.algo as an Algorithm."""
        self.algo = algo

    def run_algorithm_on_f_list(self, f_list, n_f):
        """
        Run the algorithm by sequentially querying f_list, which calls a list of n_f
        functions given an x_list of n_f inputs. Return the lists of execution paths and
        outputs.
        """

        # Create n_f copies 
        algo_list = [self.algo.get_copy() for _ in range(n_f)]

        # Initialize each algo in list
        for algo in algo_list:
            algo.initialize()

        # Step through algorithms in parallel
        x_list = [algo.get_next_x() for algo in algo_list]
        while any(x is not None for x in x_list):
            y_list = f_list(x_list)
            x_list_new = []
            for algo, x, y in zip(algo_list, x_list, y_list):
                if x is not None:
                    algo.exe_path.x.append(x)
                    algo.exe_path.y.append(y)
                    x_next = algo.get_next_x()
                    # if x_next is not None:
                    #     x_next = list(x_next.values())
                else:
                    x_next = None
                x_list_new.append(x_next)
            x_list = x_list_new

        # Store algo_list
        self.algo_list = algo_list

        # Collect exe_path_list and output_list
        exe_path_list = [algo.exe_path for algo in algo_list]
        output_list = [algo.get_output() for algo in algo_list]
        return exe_path_list, output_list

    def run_algorithm_on_f_list_independent(self, f, n_f):
        """
        Run the algorithm by querying the whole execution path first, and then 
        running f_list on it.
         
        Return the lists of execution paths and outputs.
        """

        # Create n_f copies 
        algo_list = [self.algo.get_copy() for _ in range(n_f)]

        # Initialize each algo in list
        for algo in zip(algo_list):
            algo.initialize()
            x_path = algo.params.x_path
            exe_path_new = Namespace()
            exe_path_new.x = list(x_path.unbind(dim=0))
            exe_path_new.y = f(exe_path_new.x)


    def get_exe_path_list_crop(self):
        """Return get_exe_path_crop for each algo in self.algo_list."""
        exe_path_list_crop = []
        for algo in self.algo_list:
            exe_path_crop = algo.get_exe_path_crop()
            exe_path_list_crop.append(exe_path_crop)

        return exe_path_list_crop

    def crop_exe_path_old(self, exe_path):
        """Return execution path without any Nones at end."""
        try:
            final_idx = next(i for i, x in enumerate(exe_path.x) if x==None)
        except StopIteration:
            final_idx = len(exe_path.x)

        del exe_path.x[final_idx:]
        del exe_path.y[final_idx:]
        return exe_path

