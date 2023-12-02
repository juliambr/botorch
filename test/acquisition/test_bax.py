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
from botorch.acquisition.analytic import ExpectedImprovement, PosteriorVariance
from botorch.acquisition.bax import InfoBAX
from botorch.acquisition.multi_objective.predictive_entropy_search import qMultiObjectivePredictiveEntropySearch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.test_functions.synthetic import ApproximateObjective, StyblinskiTang
from torch.quasirandom import SobolEngine

from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior

# def compute_approximated_PD(dim, bounds, approx_model, alg):
#     f_approx = ApproximateObjective(dim=dim, bounds=bounds, model=approx_model)

#     alg.initialize()

#     out_approx = alg.run_algorithm_on_f(f_approx)

#     return(out_approx)

class TestBAXPDP(BotorchTestCase):

    def test_bax_pdp(self): 
        for dtype in (torch.float, torch.double): 

            dim = 3
            bounds = torch.tensor([[-5, 5]] * dim, device=self.device, dtype=dtype)
            train_X = torch.rand(8, dim, device=self.device, dtype=dtype)
            train_Y = torch.rand(8, 1, device=self.device, dtype=dtype)
            mm = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))

            params = {"name": "MyPDP", "xs": 1, "n_points": 5, "bounds": bounds.T, "grid_size": 3}
            alg = PDPAlgorithm(params)
            alg.initialize()

            # test deterministic case 
            module = InfoBAX(model=mm, algorithm=alg, exe_path_deterministic_x=True)
            X = torch.zeros(1, dim, device=self.device, dtype=dtype)

            bax = module(X)            

    def test_bax_pdp_batch(self):
        for dtype in (torch.float, torch.double):
            dim = 2
            bounds = torch.tensor([[-5, 5]] * dim, device=self.device, dtype=dtype)
            train_X = torch.rand(8, dim, device=self.device, dtype=dtype)
            train_Y = torch.rand(8, 1, device=self.device, dtype=dtype)
            mm = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))

            params = {"name": "MyPDP", "xs": 1, "n_points": 5, "bounds": bounds.T, "grid_size": 3}
            alg = PDPAlgorithm(params)
            alg.initialize()

            # test deterministic case 
            module = InfoBAX(model=mm, algorithm=alg, exe_path_deterministic_x=True)
            X = torch.rand(3, 1, dim, device=self.device, dtype=dtype)

            bax = module(X)            
            
            self.assertEqual(bax.shape, torch.Size([3]))

    def test_info_bax_alternative(self):        
        # Paretoset
        EIG = InfoBAX(model=self.model, algorithm=self.algorithm, fixed_x_execution_path=True, n_path=1)
        # Tensor of shape (n_path, len_exe_path, d)
        # Corresponding to (n_batch, P, d)
        pareto_sets = torch.stack([torch.stack(el.x) for el in EIG.exe_path_list], dim=0)

        acq = qMultiObjectivePredictiveEntropySearch(
                    model=self.model,
                    pareto_sets=pareto_sets,
                    maximize=True
                )

        acq(self.candidate_set.unsqueeze(1))        

        candidate, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=self.bounds,
            q=1,  # Number of candidates to return
            num_restarts=1,  # Optimization restarts
            raw_samples=20,  # Samples for initialization heuristic
        )        

        print("bla")