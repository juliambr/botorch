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
from botorch.acquisition.utils import Timer, combine_data_namespaces
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model import Model

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
        self.params.num_mv_samples = getattr(params, "num_mv_samples")
        # self.params.crop = getattr(params, "crop", True)
        # self.posterior_transform = posterior_transform TODO 
        # self.set_X_pending(X_pending) TODO 

        self.set_model(model)
        self.set_algorithm(algorithm)

        exe_path_list, output_list, full_list = self.get_exe_path_and_output_samples()

        # Set self.output_list
        self.output_list = output_list
        self.exe_path_full_list = full_list

        # Set self.exe_path_list to list of full or cropped exe paths
        # if self.params.crop:
        #     self.exe_path_list = exe_path_list
        # else:
        #     
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

    def call_function_sample_list(self, x_list):
        y_list = None

        for x, query_ns in zip(x_list, self.fsl_queries):
            # Get y for a posterior function sample at x
            x = x.float()

            data = Namespace(x=self.model.train_inputs[0], y=self.model.train_targets)
            # Next round we need to combine it with query_ns (what has already been queried for that algorithm)
            comb_data = combine_data_namespaces(data, query_ns)

            model = deepcopy(self.model)

            if x is not None:
                model.set_train_data(inputs=comb_data.x, targets=comb_data.y, strict=False)
                post = self.model.posterior(x)
                y = post.rsample()[0]

                # Update query history
                if query_ns.x is None: 
                    query_ns.x = x
                    query_ns.y = y
                else: 
                    query_ns.x = torch.cat((query_ns.x, x.unsqueeze(1).T), 0)
                    query_ns.y = torch.cat((query_ns.y, y.flatten()), 0)
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
        std = posterior.variance.detach().numpy().flatten()

        # 2. Formula for entropy (standard normal)
        h_post = np.log(std) + np.log(np.sqrt(2 * np.pi)) + 0.5

        # Part 2: E[H(y_x | (D, eA))] given our different execution path samples eA

        # Dt 

        data = Namespace(x=self.model.train_inputs[0], y=self.model.train_targets)
        dim = data.x.shape[1]

        h_samp_list = []

        for exe_path in self.exe_path_list:

            # 1. Merge (D, eA)
            # exe_path.x = torch.stack(exe_path.x)
            # exe_path.y = torch.stack(exe_path.y)

            comb_data = Namespace()
            comb_data.x = torch.cat((data.x, torch.stack(exe_path.x, dim=0)), 0)
            comb_data.y = torch.cat((data.y, torch.stack(exe_path.y, dim=0).flatten()),0).unsqueeze(-1)
            comb_data.y = comb_data.y.to(dtype=torch.float64)

            model_new = SingleTaskGP(train_X=comb_data.x, train_Y=comb_data.y)
            posterior_samp = model_new.posterior(X)     

            # model_new = deepcopy(self.model)
            # model_new.set_train_data(inputs=comb_data.x, targets=comb_data.y, strict=False)
            # model_new = model_new.train()


            # model_new = SingleTaskGP(train_X=comb_data.x, train_Y=comb_data.y, input_transform=Normalize(d=dim), outcome_transform=Standardize(m=1))
            # model_new.train()
            # model_new.zero_grad()
            # mll = ExactMarginalLogLikelihood(model_new.likelihood, model_new)
            # fit_gpytorch_mll(mll);

            # model_new = deepcopy(self.model)

            # model_new.set_train_data(inputs=comb_data.x, targets=comb_data.y, strict=False) # TODO: Check re-fit? 
            # model_new.zero_grad()
            # model_new.train()

            # # 2. Computer posterior y_x | (D, eA)
            # posterior_samp = model_new.posterior(X)
            std_samp = posterior_samp.variance.detach().numpy().flatten()

            # 3. Formula for entropy 
            h_samp = np.log(std_samp) + np.log(np.sqrt(2 * np.pi)) + 0.5
            h_samp_list.append(h_samp)

        # 4. Compute mean E[(...)]
        avg_h_samp = np.mean(h_samp_list, 0)

        # Part 3: Combine the part 1 and part 2
        acq_exe = h_post - avg_h_samp

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        r"""Computes the information gain at the design points `X`.

        Approximately computes the information gain at the design points `X`,
        for both MES with noisy observations and multi-fidelity MES with noisy
        observation and trace observations.

        The implementation is inspired from the papers on multi-fidelity MES by
        [Takeno2020mfmves]_. The notation in the comments in this function follows
        the Appendix C of [Takeno2020mfmves]_.

        `num_fantasies = 1` for non-fantasized models.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design point each.
            mean_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of means.
            variance_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of variances.
            covar_mM: A
                `batch_shape x num_fantasies x (m) x (1 + num_trace_observations)`-dim
                Tensor of covariances.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X` (`num_fantasies=1` for non-fantasized models).
        """
        with Timer(f"Compute acquisition function for a batch of {len(X)} points"):
            # Compute posterior, and post given each execution path sample, for x_list
            # mu and uncertainty of y_x
            mu, std = self.model.get_post_mu_cov(X, full_cov=False)

            # Compute mean and std arrays for posterior given execution path samples
            mu_list = []
            std_list = []
            for exe_path in self.exe_path_list:
                comb_data = Namespace()
                comb_data.x = self.model.data.x + exe_path.x
                comb_data.y = self.model.data.y + exe_path.y
                # Prediction at y_x and uncertainty of y_x if execution path and data was given
                samp_mu, samp_std = self.model.gp_post_wrapper(
                    X, comb_data, full_cov=False
                )
                mu_list.append(samp_mu)
                std_list.append(samp_std)

            # Compute acq_list, the acqfunction value for each x in x_list
            if self.params.acq_str == "exe":
                acq_list = self.acq_exe_normal(std, std_list)
            elif self.params.acq_str == 'out':
                acq_list = self.acq_out_normal(std, mu_list, std_list, self.output_list)
            elif self.params.acq_str == 'is':
                acq_list = self.acq_is_normal(
                    std, mu_list, std_list, self.output_list, X
                )

        # Package and store acq_vars
        self.acq_vars = {
            "mu": mu,
            "std": std,
            "mu_list": mu_list,
            "std_list": std_list,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return acq_list
