#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for the expected information gain about the algorithm output. 

References

.. [Neiswanger2021Bax]
    Neiswanger, W., et al.,
    Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information.
    International Conference on Machine Learning, 2021.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
from copy import deepcopy
import copy
from math import log
from typing import Any, Callable, Optional

from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

import numpy as np
import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.algorithm import AlgorithmSet
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils import Timer

from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood

CLAMP_LB = 1.0e-8

class InfoBAX(AcquisitionFunction, ABC): 
    r"""Abstract base class for acquisition functions based on Max-value Entropy Search.

    This class provides the basic building blocks for constructing max-value
    entropy-based acquisition functions along the lines of [Wang2017mves]_.

    Subclasses need to implement `_sample_max_values` and _compute_information_gain`
    methods.

    :meta private:
    """

    def __init__(
        self, 
        params=None,
        model=None,
        algorithm=None
        # posterior_transform: Optional[PosteriorTransform] = None, TODO: Implement if needed 
        # X_pending: Optional[Tensor] = None
    ) -> None: 
        r"""Expected information gain (EIG) for algorithm output as acquisition function. 

        Args:
            model: A fitted single-outcome model.
            Namespace with parameters for the AcqFunction
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
        """
        super().__init__(model=model)

        params = Namespace(**params)
        self.params = Namespace()
        self.params.name = getattr(params, 'name', 'InfoBAXAcqFunction')
        self.params.n_path = getattr(params, "n_path", 10)
        self.params.exe_path_dependencies = getattr(params, "exe_path_dependencies", True)

        self.set_model(model)
        self.set_algorithm(algorithm)

        if self.params.exe_path_dependencies:
            exe_path_list, output_list, full_list = self.get_exe_path_and_output_samples()
        else: 
            exe_path_list, output_list, full_list = self.get_exe_path_and_output_samples_independent()

        # Set self.output_list
        self.output_list = output_list
        self.exe_path_full_list = full_list 
        self.exe_path_list = full_list

    def set_model(self, model):
        """Set self.model, the model underlying the acquisition function."""
        if not model:
            raise ValueError("The model input parameter cannot be None.")
        else:
            self.model = copy.deepcopy(model)

    def set_algorithm(self, algorithm):
        """Set self.algorithm for this acquisition function."""
        if not algorithm:
            raise ValueError("The algorithm input parameter cannot be None.")
        else:
            self.algorithm = algorithm.get_copy()


    def get_exe_path_and_output_samples(self):
        exe_path_list = []
        output_list = []
        with Timer(f"Sample {self.params.n_path} execution paths"):
            # Initialize model fsl
            self.fsl_queries = [Namespace(x=None, y=None) for _ in range(self.params.n_path)]

            # Run algorithm on function sample list
            f_list = self.call_function_sample_list
            algoset = AlgorithmSet(self.algorithm)
            exe_path_full_list, output_list = algoset.run_algorithm_on_f_list(
                f_list, self.params.n_path
            )

            # Get crop of each exe_path in exe_path_list
            exe_path_list = algoset.get_exe_path_list_crop()

        return exe_path_list, output_list, exe_path_full_list

    def get_exe_path_and_output_samples_independent(self):

        with Timer(f"Sample {self.params.n_path} execution paths"):
            x_path = self.algorithm.params.x_path.float()
            x_path_l = list(x_path.unbind(dim=0))
            posterior = self.model.posterior(x_path)

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.params.n_path]))
            post_samples = sampler(posterior).unbind(dim=0)        

            # Create n_f copies 
            algo_list = [self.algorithm.get_copy() for _ in range(self.params.n_path)]

            # Initialize each algo in list
            for algo, post in zip(algo_list, post_samples):
                algo.exe_path.x = x_path_l
                algo.exe_path.y = post.unbind(dim=0)

            exe_path_full_list = [algo.exe_path for algo in algo_list]
            output_list = [algo.get_output() for algo in algo_list]
            exe_path_list = exe_path_full_list

        return exe_path_list, output_list, exe_path_full_list


    def call_function_sample_list(self, x_list):
        y_list = None

        for x, query_ns in zip(x_list, self.fsl_queries):
            # Get y for a posterior function sample at x
            x = x.float()

            data = Namespace(x=self.model.train_inputs[0], y=self.model.train_targets)
            # Next round we need to combine it with query_ns (what has already been queried for that algorithm)
            # comb_data = combine_data_namespaces(data, query_ns)

            model = deepcopy(self.model)

            if x is not None:

                if query_ns.x is not None: 
                    post = model.posterior(x)
                    X_new = query_ns.x.view(-1, self.model.train_inputs[0].shape[1])
                    Y_new = query_ns.y
                    model = model.condition_on_observations(X=X_new, Y=Y_new)

                post = model.posterior(x)
                y = post.rsample()[0]

                # Update query history
                if query_ns.x is None: 
                    query_ns.x = x.unsqueeze(1).T
                    query_ns.y = y
                else: 
                    query_ns.x = torch.cat((query_ns.x, x.unsqueeze(1).T), 0)
                    query_ns.y = torch.cat((query_ns.y, y), 0)
            else:
                y = None

            if y_list is None:
                y_list = y
            else: 
                y_list = torch.cat((y_list, y), 0)
        
        return(y_list)
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute EIG at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of EIG values at the given design points `X`.
        """
        # Part 1: H(fx | AT)

        # 1. Posterior y_x | D for different x 
        posterior = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=False,
            posterior_transform=None
        )
        mu = posterior.mean
        postvar = posterior.variance.detach().numpy().flatten()
        # TODO: Probably take the root

        # 2. Formula for entropy (standard normal)
        h_post = 0.5 * np.log(2 * np.pi * postvar) + 0.5

        # Part 2: E[H(y_x | (D, eA))] given our different execution path samples eA

        # Dt 

        data = Namespace(x=self.model.train_inputs[0], y=self.model.train_targets)
        dim = data.x.shape[1]

        h_samp_list = []

        for exe_path in self.exe_path_list:

            exe_path_new = Namespace()

            # 1. Merge (D, eA)
            exe_path_new.x = torch.stack(exe_path.x).to(dtype=torch.float)
            exe_path_new.y = torch.stack(exe_path.y).to(dtype=torch.float).flatten().detach()

            model_cond = copy.deepcopy(self.model)
            post = model_cond.posterior(X)
            X_new = exe_path_new.x
            Y_new = exe_path_new.y.view(-1, 1)
            model_cond = model_cond.condition_on_observations(X=X_new, Y=Y_new)
        
            posterior_samp = model_cond.posterior(X)     

            # # 2. Computer posterior y_x | (D, eA)
            postvar_samp = posterior_samp.variance.detach().numpy().flatten()

            # 3. Formula for entropy 
            h_samp = 0.5 * np.log(2 * np.pi * postvar_samp) + 0.5
            h_samp_list.append(h_samp)

        # 4. Compute mean E[(...)]
        avg_h_samp = np.mean(h_samp_list, 0)

        # Part 3: Combine the part 1 and part 2
        acq_exe = h_post - avg_h_samp

        return(acq_exe)

