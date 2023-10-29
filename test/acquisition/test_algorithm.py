#!/usr/bin/env python3
# Copyright (c) XXXXXXXXXXXXXXXXXXXXXXX
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.acquisition.algorithm import Algorithm, AlgorithmSet, FixedPathAlgorithm, PDPAlgorithm
from botorch.acquisition.utils import compute_data_grid
from botorch.test_functions.synthetic import StyblinskiTang
from botorch.utils.testing import BotorchTestCase

class TestAlgorithm(BotorchTestCase):
    def setUp(self):
        super().setUp()
        self.test_params = {"name": "Algorithm", "x_path": []}
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            Algorithm(self.test_params)

class TestFixedPathAlgorithm(BotorchTestCase):
    def setUp(self):
        super().setUp()
        dims = 3
        dtype = torch.float64
        n_points = 10
        bounds = torch.tensor([[-5, 5]] * dims, dtype=dtype).T
        grid_size = 3
        xs=1
        grid = compute_data_grid(xs=xs,bounds=bounds,n_points=n_points,grid_size=grid_size)
        self.params = {"name": "Algorithm", "xs": xs, "x_path": grid}
        self.dims = dims

    def test_run_algorithm_on_f(self):
        f = StyblinskiTang(dim=self.dims)
        alg = FixedPathAlgorithm(self.params)
        alg.initialize()
        # For the FixedPathAlgorithm this just returns the y values on the execution path 
        alg.run_algorithm_on_f(f)

class TestPDPAlgorithm(BotorchTestCase):
    def setUp(self):
        super().setUp()
        dims = 3
        dtype = torch.float64
        n_points = 10
        bounds = torch.tensor([[-5, 5]] * dims, dtype=dtype).T
        grid_size = 3
        xs=1
        self.params = {"name": "MyPDP", "xs": xs, "n_points": n_points, "bounds": bounds, "grid_size": grid_size}
        self.dims = dims

    def test_run_algorithm_on_f(self):
        f = StyblinskiTang(dim=self.dims)
        alg = PDPAlgorithm(self.params)
        alg.initialize()
        # For the FixedPathAlgorithm this just returns the y values on the execution path 
        alg.run_algorithm_on_f(f)


class TestAlgorithmSet(TestFixedPathAlgorithm):
    def setUp(self):
        super().setUp()
        self.n_f = 3
        self.f = StyblinskiTang(dim=self.dims)
        def call_function_sample_list(x_list):
            y_list = []
            for x in x_list:
                if x is not None:
                    y = self.f(x)
                    y_list.append(y)
            return(y_list)
        self.call_function_sample_list = call_function_sample_list

    def test_run_algorithm_on_f_list(self):
        alg=FixedPathAlgorithm(self.params)
        algset = AlgorithmSet(alg)
        # Runs the same algorithm on n_f different realizations of a function 
        algset.run_algorithm_on_f_list(self.call_function_sample_list, self.n_f)
