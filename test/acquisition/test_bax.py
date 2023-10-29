#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood

import torch
from botorch.acquisition.algorithm import PDPAlgorithm
from botorch.acquisition.bax import InfoBAX

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions.synthetic import StyblinskiTang
from torch.quasirandom import SobolEngine

from botorch.utils.testing import BotorchTestCase

class TestBAX(BotorchTestCase):
    def setUp(self):
        super().setUp()
        dims = 3

        self.dtype = torch.float
        self.bounds = torch.tensor([[-5, 5]] * dims, dtype=self.dtype).T
        self.dims = dims
        self.n_samples = 40
        self.sobol = SobolEngine(dimension=self.dims, scramble=True, seed=None)

        self.f = StyblinskiTang(dim=self.dims)
        self.data = Namespace()
        X = self.sobol.draw(n=self.n_samples).to(dtype=self.dtype, device="cpu")
        self.data.x = self.bounds[0,:] + (self.bounds[1,:] - self.bounds[0,:]) * X 
        self.data.y = self.f(self.data.x).unsqueeze(0).T

        self.model = SingleTaskGP(train_X=self.data.x, train_Y=self.data.y, input_transform=Normalize(d=self.dims), outcome_transform=Standardize(m=1))
        # self.model_st = model_st.to(device=self.device, dtype=dtype)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll);

        print("Lengthscale:", self.model.covar_module.base_kernel.lengthscale)
        print("Outputscale:", self.model.covar_module.outputscale)
        print("Noise:", self.model.likelihood.noise)

        candidate_set = torch.rand(20, self.dims, device="cpu", dtype=torch.float)
        self.candidate_set = self.bounds[0,:] + (self.bounds[1,:] - self.bounds[0,:]) * candidate_set

        n_points = 10
        bounds = torch.tensor([[-5, 5]] * dims, dtype=self.dtype).T
        grid_size = 3
        xs=1
        self.params = {"name": "MyPDP", "xs": xs, "n_points": n_points, "bounds": bounds, "grid_size": grid_size}
        alg = PDPAlgorithm(self.params)
        alg.initialize()
        self.algorithm = alg

    def test_info_bax(self):
        params = {"name": "BAX", "n_path": 2, "num_mv_samples": 20}
        EIG = InfoBAX(params=params, model=self.model, algorithm=self.algorithm)
        EIG.forward(self.candidate_set)
        print("bla")