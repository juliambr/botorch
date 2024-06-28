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
from botorch.acquisition.bax import BOBAX, InfoBAX
from botorch.acquisition.multi_objective.predictive_entropy_search import qMultiObjectivePredictiveEntropySearch
from botorch.acquisition.utils import get_optimal_samples

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

            params = {"name": "MyPDP", "xs": 1, "n_points": 10, "bounds": bounds.T, "grid_size": 20}
            alg = PDPAlgorithm(params)
            alg.initialize()

            # test deterministic case 
            module = InfoBAX(model=mm, algorithm=alg, exe_path_deterministic_x=True)
            X = torch.rand(3, 1, dim, device=self.device, dtype=dtype)

            bax = module(X)            
            
            self.assertEqual(bax.shape, torch.Size([3]))

    def test_bax_pdp_subsample(self):      
        for dtype in (torch.float, torch.double): 

            dim = 3
            bounds = torch.tensor([[-5, 5]] * dim, device=self.device, dtype=dtype)
            train_X = torch.rand(8, dim, device=self.device, dtype=dtype)
            train_Y = torch.rand(8, 1, device=self.device, dtype=dtype)
            mm = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))

            params = {"name": "MyPDP", "xs": 1, "n_points": 100, "bounds": bounds.T, "grid_size": 20}
            alg = PDPAlgorithm(params)
            alg.initialize()

            # test deterministic case 
            module = InfoBAX(model=mm, algorithm=alg, exe_path_deterministic_x=True, subsample_size=5)
            X = torch.zeros(1, dim, device=self.device, dtype=dtype)

            bax = module(X)     

    def test_bobax_pdp(self):      
        for dtype in (torch.float, torch.double): 

            dim = 3
            bounds = torch.tensor([[-5, 5]] * dim, device=self.device, dtype=dtype)
            train_X = torch.rand(8, dim, device=self.device, dtype=dtype)
            train_Y = torch.rand(8, 1, device=self.device, dtype=dtype)
            mm = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))

            params = {"name": "MyPDP", "xs": 1, "n_points": 20, "bounds": bounds.T, "grid_size": 10}
            alg = PDPAlgorithm(params)
            alg.initialize()

            num_samples = 8

            optimal_inputs, optimal_outputs = get_optimal_samples(
                mm,
                bounds=bounds.T,
                num_optima=num_samples
            )

            # test deterministic case 
            module = BOBAX(model=mm, algorithm=alg, optimal_inputs=optimal_inputs, maximize=False)
            X = torch.zeros(1, dim, device=self.device, dtype=dtype)

            bax = module(X)     

    def test_bobax_pdp_subsample(self):      
        for dtype in (torch.float, torch.double): 

            dim = 3
            bounds = torch.tensor([[-5, 5]] * dim, device=self.device, dtype=dtype)
            train_X = torch.rand(8, dim, device=self.device, dtype=dtype)
            train_Y = torch.rand(8, 1, device=self.device, dtype=dtype)
            mm = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))

            params = {"name": "MyPDP", "xs": 1, "n_points": 20, "bounds": bounds.T, "grid_size": 10}
            alg = PDPAlgorithm(params)
            alg.initialize()

            num_samples = 8

            optimal_inputs, optimal_outputs = get_optimal_samples(
                mm,
                bounds=bounds.T,
                num_optima=num_samples
            )

            # test deterministic case 
            module = BOBAX(model=mm, algorithm=alg, optimal_inputs=optimal_inputs, maximize=False, subsample_size=3)
            X = torch.zeros(1, dim, device=self.device, dtype=dtype)

            bax = module(X)     
      
    

    def test_bobax_pdp_batch(self):      
        for dtype in (torch.float, torch.double): 

            dim = 3
            bounds = torch.tensor([[-5, 5]] * dim, device=self.device, dtype=dtype)
            train_X = torch.rand(8, dim, device=self.device, dtype=dtype)
            train_Y = torch.rand(8, 1, device=self.device, dtype=dtype)
            mm = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))

            params = {"name": "MyPDP", "xs": 1, "n_points": 10, "bounds": bounds.T, "grid_size": 20}
            alg = PDPAlgorithm(params)
            alg.initialize()

            num_samples = 8

            optimal_inputs, optimal_outputs = get_optimal_samples(
                mm,
                bounds=bounds.T,
                num_optima=num_samples
            )

            # test deterministic case 
            module = BOBAX(model=mm, algorithm=alg, optimal_inputs=optimal_inputs, maximize=False)

            candidate_set = torch.rand(50, dim, device="cpu")
            candidate_set = bounds[:,0] + (bounds[:,1] - bounds[:,0]) * candidate_set
            candidate_set = candidate_set.unsqueeze(1)

            bobax = module(candidate_set)            
            
            self.assertEqual(bobax.shape, torch.Size([50]))

    def test_bobax_pdp_acq_optim(self):    

        for dim in [2, 3, 5, 8]:  
            for dtype in (torch.float, torch.double): 

                # dim = 5
                bounds = torch.tensor([[-5, 5]] * dim, device=self.device, dtype=dtype)
                train_X = torch.rand(8, dim, device=self.device, dtype=dtype)
                train_Y = torch.rand(8, 1, device=self.device, dtype=dtype)
                mm = SingleTaskGP(train_X=train_X, train_Y=train_Y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))

                params = {"name": "MyPDP", "xs": 1, "n_points": 100, "bounds": bounds.T, "grid_size": 20}
                alg = PDPAlgorithm(params)
                alg.initialize()

                num_samples = 8

                optimal_inputs, optimal_outputs = get_optimal_samples(
                    mm,
                    bounds=bounds.T,
                    num_optima=num_samples
                )

                # test deterministic case 
                module = BOBAX(model=mm, algorithm=alg, optimal_inputs=optimal_inputs, maximize=False)

                candidate, acq_value = optimize_acqf(
                    acq_function=module,
                    bounds=bounds.T,
                    q=1,
                    num_restarts=10,
                    raw_samples=512,
                    options={"with_grad": False},
                )