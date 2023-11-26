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

from abc import ABC
from argparse import Namespace
from copy import deepcopy
import copy
from math import log, pi
from typing import Any
import warnings
import logging
from botorch.acquisition.analytic import AnalyticAcquisitionFunction

import numpy as np
import torch
from botorch.acquisition.algorithm import AlgorithmSet
from botorch.sampling import SobolQMCNormalSampler, DeterministicSampler
import time 

CLAMP_LB = 1.0e-8

class InfoBAX(AnalyticAcquisitionFunction, ABC): 
    r"""Base class for Information-based Bayesian Algorithm Execution. 
     
    This class provides an implementation of infill-criterion introduced in [Neiswanger2021Bax]_.
    Note that this is an implementation of 3.1 (EIG for Execution Path) in [Neiswanger2021Bax]_ only. 
    Please verify the assumptions of whether this is suitable depending on the algorithm chosen. 
    """

    def __init__(
        self, 
        model=None,
        algorithm=None,
        fixed_x_execution_path: bool = False,
        n_path: int = 1,
        # posterior_transform: Optional[PosteriorTransform] = None, TODO: Implement if needed 
    ) -> None: 
        r"""Expected information gain (EIG) for execution path. 

        Args:
            model: A fitted single-outcome model.
            fixed_x_execution_path: True if the x-values of the execution path sequence is deterministic.

            params: Parameters of the execution 
            Namespace with parameters for the AcqFunction
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
        """
        super().__init__(model=model)

        self.params = Namespace()
        self.params.fixed_x_execution_path = fixed_x_execution_path
        self.model = copy.deepcopy(model)
        self.algorithm = copy.deepcopy(algorithm)

        if fixed_x_execution_path and n_path > 1: 
            warnings.warn('n_path is always 1 if fixed_x_execution_path is True to enhance computational efficiency. ', UserWarning)
            self.params.n_path = 1
        else:
            self.params.n_path = n_path

        logging.info('Sampling of execution paths started ...')

        tstart = time.time()

        if self.params.fixed_x_execution_path:
            exe_path_list, output_list, full_list = self.get_exe_path_fixed_x()
        else: 
            exe_path_list, output_list, full_list = self.get_exe_path_and_output_samples()

        logging.info('Execution path sampling complete. Time elapsed: %.2f seconds' % (time.time() - tstart))
        
        self.output_list = output_list
        self.exe_path_full_list = full_list 
        self.exe_path_list = exe_path_list

    def get_exe_path_fixed_x(self):

        x_path = self.algorithm.params.x_path.float()
        x_path_l = list(x_path.unbind(dim=0))
        # posterior = self.model.posterior(x_path)

        # sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.params.n_path]))
        # post_samples = sampler(posterior).unbind(dim=0)        

        # Create n_f copies 
        algo_list = [self.algorithm.get_copy() for _ in range(self.params.n_path)]

        # Initialize each algo in list
        for algo in algo_list: # for algo, post in zip(algo_list, post_samples):
            algo.exe_path.x = x_path_l
            algo.exe_path.y = None # post.unbind(dim=0)

        exe_path_full_list = None # [algo.exe_path for algo in algo_list]
        output_list = None # [algo.get_output() for algo in algo_list]
        exe_path_list = [algo.exe_path for algo in algo_list]

        return exe_path_list, output_list, exe_path_full_list

    def get_exe_path_and_output_samples(self):
        exe_path_list = []
        output_list = []
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
    

    def forward(self, X: torch.Tensor, min_var: float = 1e-12) -> torch.Tensor:
        r"""Compute EIG at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of EIG values at the given design points `X`.
        """

        # Part 1: H(y_x | A_T) for different values x

        # 1. Compute posterior mean and sd 
        _, sigma = self._mean_and_sigma(X) # Note that this returns sd not var 

        # 2. Formula for entropy (standard normal)
        #    H(y_x | A_T)   = 0.5 * log(2 * pi * sigma^2) + 0.5 = 
        #                   = 0.5 * log(2 * pi) + log(sigma) + 0.5
        h_post = torch.log(sigma) + 0.5 * log(2 * pi) +  0.5

        # Part 2: E[H(y_x | (A_T, e_A))] given our different execution path samples e_A

        h_samp_list = []

        tstart = time.time()

        logging.info('Computation of posterior given archive and execution path ...')

        for exe_path in self.exe_path_list:

            # 1. y | (A_t, x, e_A): Condition the posterior process y_x | (A_t, x) additionally on e_A

            # The data we need to condition on 
            X_new = torch.stack(exe_path.x).to(dtype=torch.float64)
            # Note: The computation of the entropy does not depend on the y values of the execution path
            #       This is because the entropy only depends on the posterior variance, which only depends on X and not on Y for a GP
            #       Note that the mean does, however the entropy does not need the mean. 
            #       Therefore, we only fantasize some deterministic values. 
            #       See the sample below for some evidence 

            model_fantasized = copy.deepcopy(self.model)
            sampler_det = DeterministicSampler(sample_shape=torch.Size([]))
            model_fantasized = model_fantasized.fantasize(X_new, sampler_det)

            # y | (A_t, x, e_A)
            posterior_fantasized = model_fantasized.posterior(X)     
            # mean = posterior_fantasized.mean.squeeze(-2).squeeze(-1) # TODO: Do we need this one?
            var_fantasized = posterior_fantasized.variance.squeeze()
            sigma_fantasized = var_fantasized.clamp_min(min_var).sqrt()

            # The below holds: 
            # sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.params.n_path]))
            # model_fantasized2 = copy.deepcopy(self.model)
            # model_fantasized2 = model_fantasized2.fantasize(X_new, sampler)
            # posterior_fantasized2 = model_fantasized2.posterior(X)     
            # var_fantasized2 = posterior_fantasized2.variance
            # torch.equal(var_fantasized, var_fantasized2)

            # 2. Formula for entropy 
            h_samp= torch.log(sigma_fantasized) + 0.5 * log(2 * pi) +  0.5
            h_samp_list.append(h_samp)

        message = 'Elapsed: %.2f seconds' % (time.time() - tstart)
        logging.info(message)

        # 4. Compute mean E[(...)]
        avg_h_samp = torch.mean(torch.stack(h_samp_list), dim=0)

        # Part 3: Combine the part 1 and part 2
        acq_exe = h_post - avg_h_samp

        return(acq_exe)

