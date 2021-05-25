"""
    server-side Flimma project to aggregate the local parameters from the clients

    Copyright 2021 Olga Zolotareva, Reza NasiriGerdeh, and Mohammad Bakhtiari. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from hyfed_server.project.hyfed_server_project import HyFedServerProject
from hyfed_server.util.hyfed_steps import HyFedProjectStep
from hyfed_server.util.status import ProjectStatus
from hyfed_server.util.utils import client_parameters_to_list

from flimma_server.util.flimma_steps import FlimmaProjectStep
from flimma_server.util.flimma_parameters import FlimmaGlobalParameter, FlimmaLocalParameter, FlimmaProjectParameter

import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from scipy import linalg
import statsmodels.api as sm
import pickle
from scipy.special import digamma, polygamma
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from copy import copy
from bioinfokit import visuz

class FlimmaServerProject(HyFedServerProject):
    """ Server side of Flimma project

    Attributes
    ----------
    global_sample_count: int
    n_clients: int
    cohort_effects: list
    global_features: list
    cohort_names: list
    n_features: int
    min_per_group_num_samples: int
    genes_passed_total_count: list
    gene_name_list: list
    cov_coefficient: list
    sigma: numpy.ndarray
        estimated residual standard deviations
    degree_of_freedom: numpy.ndarray
    """
    large_n = 10
    min_prop = 0.7
    tol = 1e-14

    def __init__(self, creation_request, project_model):
        """ Initialize Flimma project attributes based on the values set by the coordinator """
        # initialize base project
        super().__init__(creation_request, project_model)

        try:
            # save project (hyper-)parameters in the project model and initialize the project
            flimma_model_instance = project_model.objects.get(id=self.project_id)

            for attr, val in FlimmaProjectParameter.__dict__.items():
                if not attr.startswith('__'):
                    temp_val = creation_request.data[val]
                    setattr(flimma_model_instance, val, temp_val)
                    setattr(self, val, temp_val)
                    print(attr, val, temp_val)

            self.contrast1_list = [variable.strip() for variable in self.group1.split(',')]
            self.contrast2_list = [variable.strip() for variable in self.group2.split(',')]
            self.variables = self.contrast1_list + self.contrast2_list
            self.confounders = self.confounders.strip().split(",")

            # result directory
            self.result_dir = flimma_model_instance.result_dir = "flimma_server/result"

            # save the model
            flimma_model_instance.save()
            logger.debug(f"Project {self.project_id}: Flimma specific attributes initialized!")

            # global attributes
            self.global_sample_count = 0
            self.n_clients = 0
            self.cohort_effects = []
            self.gene_name_list, self.cohort_names = [], []
            self.n_features = 0
            self.min_per_group_num_samples = None
            self.genes_passed_total_count = None
            self.gene_name_list = []
            self.beta = None
            self.cov_coefficient = []
            self.sse = None
            self.variance = None
            self.sigma = np.array([])
            self.degree_of_freedom = np.array([])
            self.mean_count = None
            self.mean_log_count = None
            self.py_lowess = None
            self.contrast_matrix = None
            self.std_unscaled = None
            self.df_total = None
            self.results = {}
            self.table = None

        except Exception as model_exp:
            logger.error(f'{bcolors.FAIL}Project {self.project_id}: {model_exp}')
            self.project_failed()

    # ############### Project step functions ####################
    def init_step(self):
        """ initialize Flimma server project """

        try:
            # get the sample counts from the clients and compute global sample count, which is used in the next steps
            sample_counts = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.SAMPLE_COUNT)
            self.global_sample_count = np.sum(sample_counts)

            # tell clients to go to the PREPARE_INPUTS step
            self.set_step(FlimmaProjectStep.PREPARE_INPUTS)

        except Exception as init_exception:
            logger.error(f'{bcolors.FAIL}Project {self.project_id}: {init_exception}')
            self.project_failed()

    def prepare_inputs_step(self):
        """ Collect feature names and cohort names from client, send back shared features and the list of cohort names"""

        try:
            # collect cohort names ('Cohort_'+username)
            self.cohort_names = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.COHORT_NAME)
            self.cohort_effects = sorted(self.cohort_names)[:-1]  # will add confounders for all but one cohorts

            # collect features from clients and keep only features shared by all clients
            feature_lists = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.FEATURES)
            # for f in feature_lists:
            #     print(len(f))
            shared_features = set(feature_lists[0])
            for feature_list in feature_lists[1:]:
                shared_features = shared_features.intersection(set(feature_list))
            self.gene_name_list = sorted(list(shared_features))
            self.n_features = len(self.gene_name_list)

            # testprints
            print("#############")
            print("Total samples:", self.global_sample_count)
            print("Shared features:", self.gene_name_list[:3], "...", len(self.gene_name_list), "features")
            print("Joined cohorts:", self.cohort_names)
            print("Cohort effects added:", self.cohort_effects)
            print("############")

            # send to clients 1) shared features 2) the list of cohort names
            self.global_parameters[FlimmaGlobalParameter.FEATURES] = self.gene_name_list
            self.global_parameters[FlimmaGlobalParameter.COHORT_EFFECTS] = self.cohort_effects

            # tell clients to go to the CPM_CUTOFF step
            self.set_step(FlimmaProjectStep.CPM_CUTOFF)

        except Exception as init_exception:
            logger.error(f'{bcolors.FAIL}Project {self.project_id} prepare_inputs_step() failed: {init_exception}')
            self.project_failed()

    def compute_cpm_cutoff_step(self):
        """ Collect local parameters and for CPM cutoff. Compute global CPM cutoff and send it to clients. """

        try:
            ### collect local parameters for CPM cutoff as in FilteByExprs()
            # the number of samples per group

            n_samples_per_group = client_parameters_to_list(self.local_parameters,
                                                            FlimmaLocalParameter.N_SAMPLES_PER_GRPOUP)
            self.total_num_samples = np.sum(n_samples_per_group, axis=0)
            
            # total read counts per gene
            clients_count_per_feature = client_parameters_to_list(self.local_parameters,
                                                                  FlimmaLocalParameter.TOTAL_COUNT_PER_FEATURE)
            self.total_count_per_feature = np.sum(clients_count_per_feature, axis=0)
            
            # # lib.sizes
            # clients_lib_sizes = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.LIB_SIZES)
            # total_lib_sizes = np.concatenate(clients_lib_sizes)
            #
            # # define min allowed number of samples
            # self.min_per_group_num_samples = np.min([x for x in total_num_samples if x > 0])
            #
            # if self.min_per_group_num_samples > self.large_n:
            #     self.min_per_group_num_samples = self.large_n + (
            #             self.min_per_group_num_samples - self.large_n) * self.min_prop
            # print("Min. sample size:", self.min_per_group_num_samples)
            #
            # self.genes_passed_total_count = \
            #     np.where(total_count_per_feature >= (self.min_per_group_num_samples - self.tol))[0]
            # print("Features in total:",len(total_count_per_feature))
            # print("features passed total count cutoff:", len(self.genes_passed_total_count))
            #
            # median_lib_size = np.median(total_lib_sizes)
            # CPM_cutoff = self.min_count / median_lib_size * 1e6
            # print("median lib.size:", median_lib_size, "\nCPM_cutoff:", CPM_cutoff)
            #
            # # send to clients 1) passed features
            # # 2) the list of cohort variables to add (all but 1st client)
            # self.global_parameters[FlimmaGlobalParameter.CPM_CUTOFF] = CPM_cutoff
            #
            self.set_step(FlimmaProjectStep.LIB_SIZES)

        except Exception as io_exception:
            logger.error(f'{bcolors.FAIL}Project {self.project_id} compute_cpm_cutoff_step() failed: {io_exception}')
            self.project_failed()

    def get_lib_sizes_step(self):
        try:
            # lib.sizes
            clients_lib_sizes = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.LIB_SIZES)
            total_lib_sizes = np.concatenate(clients_lib_sizes)

            # define min allowed number of samples
            self.min_per_group_num_samples = np.min([x for x in self.total_num_samples if x > 0])

            if self.min_per_group_num_samples > self.large_n:
                self.min_per_group_num_samples = self.large_n + (
                        self.min_per_group_num_samples - self.large_n) * self.min_prop
            print("Min. sample size:", self.min_per_group_num_samples)

            self.genes_passed_total_count = \
                np.where(self.total_count_per_feature >= (self.min_per_group_num_samples - self.tol))[0]
            print("Features in total:", len(self.total_count_per_feature))
            print("features passed total count cutoff:", len(self.genes_passed_total_count))

            median_lib_size = np.median(total_lib_sizes)
            CPM_cutoff = self.min_count / median_lib_size * 1e6
            print("median lib.size:", median_lib_size, "\nCPM_cutoff:", CPM_cutoff)

            # send to clients 1) passed features
            # 2) the list of cohort variables to add (all but 1st client)
            self.global_parameters[FlimmaGlobalParameter.CPM_CUTOFF] = CPM_cutoff

            self.set_step(FlimmaProjectStep.APPLY_CPM_CUTOFF)

        except Exception as io_exception:
            logger.error(f'{bcolors.FAIL}Project {self.project_id} compute_cpm_cutoff_step() failed: {io_exception}')
            self.project_failed()

    def apply_cpm_cutoff_step(self):
        """Apply CPM cutoff"""
        try:
            clients_cpm_cutoff_sample_count = \
                client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.CPM_CUTOFF_SAMPLE_COUNT)
            total_cpm_cutoff_sample_count = np.zeros(len(self.gene_name_list), dtype="int") + np.sum(
                clients_cpm_cutoff_sample_count, axis=0)

            genes_passed_cpm_cutoff = \
                np.where(total_cpm_cutoff_sample_count > self.min_per_group_num_samples - self.tol)[0]
            print(f"features passed CPM cutoff: {len(genes_passed_cpm_cutoff)}")
            self.gene_name_list = np.array(self.gene_name_list)[sorted(
                list(set(genes_passed_cpm_cutoff).intersection(set(self.genes_passed_total_count))))].tolist()
            print(f"features passed both cutoffs: {len(self.gene_name_list)}")
            self.global_parameters[FlimmaGlobalParameter.GENES_NAME_LIST] = self.gene_name_list

            self.set_step(FlimmaProjectStep.COMPUTE_NORM_FACTORS)

        except Exception as io_exception:
            logger.error(f'{bcolors.FAIL}Project {self.project_id} apply_cpm_cutoff_step() failed: {io_exception}')
            self.project_failed()

    def compute_norm_factors_step(self):
        try:
            clients_upper_quartiles = client_parameters_to_list(self.local_parameters,
                                                                FlimmaLocalParameter.UPPER_QUARTILE)
            clients_lib_sizes = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.UPDATED_LIB_SIZES)
            lib_sizes = np.concatenate(clients_lib_sizes, axis=None)
            upper_quartiles = np.concatenate(clients_upper_quartiles, axis=None)

            quart_to_lib_size = upper_quartiles / lib_sizes
            self.global_parameters[FlimmaGlobalParameter.F] = np.exp(np.mean(np.log(quart_to_lib_size)))
            self.set_step(FlimmaProjectStep.LINEAR_REGRESSION)
        except Exception as io_exception:
            logger.error(f'Project {self.project_id} () failed: {io_exception}')
            self.project_failed()

    def linear_regression_step(self):
        try:
            clients_xt_x = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.XT_X_MATRIX)
            clients_xt_y = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.XT_Y_MATRIX)
            global_xt_x = np.sum(clients_xt_x, axis=0)
            global_xt_y = np.sum(clients_xt_y, axis=0)
            k, n = self.get_k_n()
            self.beta = np.zeros((n, k))
            rank = np.ones(n) * k
            self.std_unscaled = np.zeros((n, k))

            for i in range(0, n):
                inv_xt_x = linalg.inv(global_xt_x[i, :, :])
                self.beta[i, :] = inv_xt_x @ global_xt_y[i, :]
                self.std_unscaled[i, :] = np.sqrt(np.diag(inv_xt_x))
            self.global_parameters[FlimmaGlobalParameter.BETA] = self.beta

        except Exception as io_exception:
            logger.error(f'{bcolors.FAIL}Project {self.project_id} linear_regression_step() failed: {io_exception}')
            self.project_failed()

    def get_k_n(self):
        k = len(self.variables) + len(self.confounders) + len(self.cohort_names) - 1
        n = len(self.gene_name_list)
        return k, n

    def sse_step(self):
        try:
            clients_sample_count = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.SAMPLE_COUNT)
            self.global_sample_count = np.sum(clients_sample_count, axis=0)
            clients_cov = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.COVARIANCE_COEFFICIENT)
            total_cov = np.sum(clients_cov, axis=0)
            clients_sse = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.SSE)
            self.sse = np.sum(clients_sse, axis=0)
            self.cov_coefficient = linalg.inv(total_cov)
            k, n = self.get_k_n()
            # estimated residual variance
            self.variance = self.sse / (self.global_sample_count - k)

            # estimated residual standard deviations
            self.sigma = np.sqrt(self.variance)

            # degrees of freedom
            self.degree_of_freedom = np.ones(n) * (self.global_sample_count - k)

        except Exception as io_exception:
            logger.error(f'{bcolors.FAIL}Project {self.project_id} sse_step() failed: {io_exception}')
            self.project_failed()

    def mean_log_count_step(self, weighted):
        try:
            clients_log_count = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.LOG_COUNT)
            total_log_count = np.sum(clients_log_count, axis=0)
            clients_log_count_conversion = client_parameters_to_list(self.local_parameters,
                                                                     FlimmaLocalParameter.LOG_COUNT_CONVERSION)
            total_log_count_conversion = np.sum(clients_log_count_conversion, axis=0)
            total_log_count_conversion = total_log_count_conversion / self.global_sample_count - 6 * np.log2(10)
            self.mean_count = total_log_count / self.global_sample_count
            self.mean_log_count = self.mean_count + total_log_count_conversion

            delta = (max(self.mean_log_count) - min(self.mean_log_count)) * 0.01
            lowess = sm.nonparametric.lowess
            self.py_lowess = lowess(self.sigma ** 0.5, self.mean_log_count, frac=0.5, delta=delta, return_sorted=True,
                                    is_sorted=False)
            
        except Exception as io_exception:
            logger.error(
                f'{bcolors.FAIL}Project {self.project_id} mean_log_count_step(weighted={weighted}) failed: {io_exception}')
            self.project_failed()

    def make_contrasts(self, contrast_list):
        '''Creates contrast matrix given design matrix and pairs or columns to compare.
        For example:
        contrasts = [([A],[B]),([A,B],[C,D])] defines two contrasts:\n
        A-B and (A and B) - (C and D).'''
        df = {}
        conditions = self.variables + self.confounders + self.cohort_names[0:-1]
        for contrast in contrast_list:
            group1, group2 = contrast
            for name in group1 + group2:
                if name not in conditions:
                    print(name, "not found in the design matrix.")
                    exit(1)
            contrast_name = "".join(map(str, group1)) + "_vs_" + "".join(map(str, group2))
            series = pd.Series(data=np.zeros(len(conditions)), index=conditions)
            series[group1] = 1
            series[group2] = -1
            df[contrast_name] = series

        self.contrast_matrix = pd.DataFrame.from_dict(df).values

    def fit_contrasts(self):
        n_coef = self.cov_coefficient.shape[1]
        #	Correlation matrix of estimable coefficients
        #	Test whether design was orthogonal
        if not np.any(self.cov_coefficient):
            print("no coefficient correlation matrix found in fit - assuming orthogonal")
            correlation_matrix = np.identity(n_coef)
            orthog = True
        else:
            print("coefficient correlation matrix is found")
            correlation_matrix = self.cov2cor()
            print("cov2cor() is called")
            if correlation_matrix.shape[0] * correlation_matrix.shape[1] < 2:
                orthog = True
            else:
                if np.sum(np.abs(np.tril(correlation_matrix, k=-1))) < 1e-12:
                    orthog = True
                else:
                    orthog = False

        #	Replace NA coefficients with large (but finite) standard deviations
        #	to allow zero contrast entries to clobber NA coefficients.
        if np.any(np.isnan(self.beta)):
            print("Replace NA coefficients with large (but finite) standard deviations")
            np.nan_to_num(self.beta, nan=0)
            np.nan_to_num(self.std_unscaled, nan=1e30)

        self.beta = self.beta.dot(self.contrast_matrix)
        # New covariance coefficiets matrix
        self.cov_coefficient = self.contrast_matrix.T.dot(self.cov_coefficient).dot(self.contrast_matrix)
        if orthog:
            self.std_unscaled = np.sqrt((self.std_unscaled ** 2).dot(self.contrast_matrix ** 2))
        else:
            n_genes = self.beta.shape[0]
            U = np.ones((n_genes, self.contrast_matrix.shape[1]))  # genes x contrasts
            o = np.ones(n_coef)
            R = np.linalg.cholesky(correlation_matrix).T
            for i in range(0, n_genes):
                RUC = R @ (self.std_unscaled[i,] * self.contrast_matrix.T).T
                U[i,] = np.sqrt(o @ RUC ** 2)
            self.std_unscaled = U

    def cov2cor(self):
        cor = np.diag(self.cov_coefficient) ** -0.5 * self.cov_coefficient
        cor = cor.T * np.diag(self.cov_coefficient) ** -0.5
        np.fill_diagonal(cor, 1)
        return cor

    # ######## eBayes
    def trigamma(self, x):
        return polygamma(1, x)

    def psigamma(self, x, deriv=2):
        return polygamma(deriv, x)

    def trigammaInverse(self, x):
        if not hasattr(x, '__iter__'):
            x_ = np.array([x])
        for i in range(0, x_.shape[0]):
            if np.isnan(x_[i]):
                x_[i] = np.NaN
            elif x > 1e7:
                x_[i] = 1. / np.sqrt(x[i])
            elif x < 1e-6:
                x_[i] = 1. / x[i]
        # Newton's method
        y = 0.5 + 1.0 / x_
        for i in range(0, 50):
            tri = self.trigamma(y)
            dif = tri * (1.0 - tri / x_) / self.psigamma(y, deriv=2)
            y = y + dif
            if np.max(-dif / y) < 1e-8:  # tolerance
                return y

        print("Warning: Iteration limit exceeded")
        return y

    def moderatedT(self, covariate=False, robust=False, winsor_tail_p=(0.05, 0.1)):

        # var,df_residual,coefficients,stdev_unscaled,
        self.squeeze_var(covariate=covariate, robust=robust, winsor_tail_p=winsor_tail_p)

        self.results["s2_prior"] = self.results["var_prior"]
        self.results["s2_post"] = self.results["var_post"]
        del self.results["var_prior"]
        del self.results["var_post"]

        self.results["t"] = self.beta / self.std_unscaled
        self.results["t"] = self.results["t"].T / np.sqrt(self.results["s2_post"])
        self.df_total = self.degree_of_freedom + self.results["df_prior"]
        df_pooled = sum(self.degree_of_freedom)
        self.df_total = np.minimum(self.df_total, df_pooled)  # component-wise min

        self.results["p_value"] = 2 * t.cdf(-np.abs(self.results["t"]), df=self.df_total)
        self.results["p_value"] = self.results["p_value"].T
        self.results["t"] = self.results["t"].T

    def squeeze_var(self, covariate=False, robust=False, winsor_tail_p=(0.05, 0.1)):
        '''Estimates df and var priors and computes posterior variances.'''
        if robust:
            # TBD fitFDistRobustly()
            print("Set robust=False.")
            return
        else:
            var_prior, df_prior = self.fitFDist(covariate=covariate)

        if np.isnan(df_prior):
            print("Error: Could not estimate prior df")
            return

        var_post = self.posterior_var(var_prior=var_prior, df_prior=df_prior)
        self.results = {"df_prior": df_prior, "var_prior": var_prior, "var_post": var_post}

    def fitFDist(self, covariate=False):
        '''Given x (sigma^2) and df1 (degree_of_freedom), 
        fits x ~ scale * F(df1,df2) and returns 
        estimated df2 and scale (s0^2)'''
        
        if covariate:
            # TBD
            print("Set covariate=False.")
            return

        # Avoid zero variances
        variances = [max(var, 0) for var in self.variance]
        median = np.median(variances)
        if median == 0:
            print("Warning: More than half of residual variances are exactly zero: eBayes unreliable")
            median = 1
        else:
            if 0 in variances:
                print("Warning: Zero sample variances detected, have been offset (+1e-5) away from zero")

        variances = [max(var, 1e-5 * median) for var in variances]
        z = np.log(variances)
        e = z - digamma(self.degree_of_freedom * 1.0 / 2) + np.log(self.degree_of_freedom * 1.0 / 2)
        emean = np.nanmean(e)
        evar = np.nansum((e - emean) ** 2) / (len(variances) - 1)

        # Estimate scale and df2
        evar = evar - np.nanmean(self.trigamma(self.degree_of_freedom * 1.0 / 2))
        if evar > 0:
            df2 = 2 * self.trigammaInverse(evar)
            s20 = np.exp(emean + digamma(df2 * 1.0 / 2) - np.log(df2 * 1.0 / 2))
        else:
            df2 = np.Inf
            s20 = np.exp(emean)

        return s20, df2

    def posterior_var(self, var_prior=np.ndarray([]), df_prior=np.ndarray([])):
        '''.squeezeVar()'''
        var = self.variance
        df = self.degree_of_freedom
        ndxs = np.argwhere(np.isfinite(var)).reshape(-1)
        # if not infinit vars
        if len(ndxs)==len(var): #np.isinf(df_prior).any():
            return (df*var + df_prior*var_prior) / (df+df_prior) # var_post  
        #For infinite df.prior, set var_post = var_prior
        var_post = np.repeat(var_prior,len(var))
        for ndx in ndxs:
            var_post[ndx] = (df[ndx]*var[ndx] + df_prior*var_prior)/(df[ndx]+df_prior)
        return var_post

    def tmixture_matrix(self, var_prior_lim=False, proportion=0.01):
        tstat = self.results["t"]
        std_unscaled = self.std_unscaled
        df_total = self.df_total
        ncoef = self.results["t"].shape[1]
        v0 = np.zeros(ncoef)
        for j in range(0, ncoef):
            v0[j] = self.tmixture_vector(tstat[:, j], std_unscaled[:, j], df_total, proportion, var_prior_lim)
        return v0

    def tmixture_vector(self, tstat, std_unscaled, df, proportion, var_prior_lim):
        ngenes = len(tstat)

        # Remove missing values
        notnan_ndx = np.where(~np.isnan(tstat))[0]
        if len(notnan_ndx) < ngenes:
            tstat = tstat[notnan_ndx]
            std_unscaled = std_unscaled[notnan_ndx]
            df = df[notnan_ndx]

        # ntarget t-statistics will be used for estimation

        ntarget = int(np.ceil(proportion / 2 * ngenes))
        if ntarget < 1:  #
            return

            # If ntarget is v small, ensure p at least matches selected proportion
        # This ensures ptarget < 1
        p = np.maximum(ntarget * 1.0 / ngenes, proportion)

        # Method requires that df be equal
        tstat = abs(tstat)
        MaxDF = np.max(df)
        i = np.where(df < MaxDF)[0]
        if len(i) > 0:
            TailP = t.logcdf(tstat[i],df=df[i])  
            # PT: CDF of t-distribution: pt(tstat[i],df=df[i],lower.tail=FALSE,log.p=TRUE)
            # QT - qunatile funciton - returns a threshold value x
            # below which random draws from the given CDF would fall p percent of the time. [wiki]
            tstat[i] = t.ppf(np.exp(TailP), df=MaxDF)  # QT: qt(TailP,df=MaxDF,lower.tail=FALSE,log.p=TRUE)
            df[i] = MaxDF

        # Select top statistics
        order = tstat.argsort()[::-1][:ntarget]  # TBD: ensure the order is decreasing
        tstat = tstat[order]
        v1 = std_unscaled[order] ** 2

        # Compare to order statistics
        rank = np.array(range(1, ntarget + 1))
        p0 = 2 * t.sf(tstat, df=MaxDF)  # PT
        ptarget = ((rank - 0.5) / ngenes - (1.0 - p) * p0) / p
        v0 = np.zeros(ntarget)
        pos = np.where(ptarget > p0)[0]
        if len(pos) > 0:
            qtarget = -t.ppf(ptarget[pos] / 2, df=MaxDF)  # qt(ptarget[pos]/2,df=MaxDF,lower.tail=FALSE)
            v0[pos] = v1[pos] * ((tstat[pos] / qtarget) ** 2 - 1)

        if var_prior_lim[0] and var_prior_lim[1]:
            v0 = np.minimum(np.maximum(v0, var_prior_lim[0]), var_prior_lim[1])

        return np.mean(v0)

    def b_stat(self, std_coef_lim=np.array([0.1, 4]), proportion=0.01):
        var_prior_lim = std_coef_lim ** 2 / np.median(self.results["s2_prior"])
        # print("Limits for var.prior:",var_prior_lim)
        
        self.results["var_prior"] = self.tmixture_matrix(proportion=0.01, var_prior_lim=var_prior_lim)

        nan_ndx = np.argwhere(np.isnan(self.results["var_prior"]))
        if len(nan_ndx) > 0:
            self.results["var.prior"][nan_ndx] < - 1.0 / self.results["s2_prior"]
            print("Warning: Estimation of var.prior failed - set to default value")
        r = np.outer(np.ones(self.results["t"].shape[0]), self.results["var_prior"])
        r = (self.std_unscaled ** 2 + r) / self.std_unscaled ** 2
        t2 = self.results["t"] ** 2

        valid_df_ndx = np.where(self.results["df_prior"] <= 1e6)[0]
        if len(valid_df_ndx) < len(self.results["df_prior"]):
            print("Large (>1e6) priors for DF:", len(valid_df_ndx))
            kernel = t2 * (1 - 1.0 / r) / 2
            for i in valid_df_ndx:
                kernel[i] = (1 + self.df_total[i]) / 2 * np.log(
                    (t2[i, :].T + self.df_total[i]) / ((t2[i, :] / r[i, :]).T + self.df_total[i]))
        else:
            kernel = (1 + self.df_total) / 2 * np.log((t2.T + self.df_total) / ((t2 / r).T + self.df_total))

        self.results["lods"] = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel.T

    def top_table_t(self, adjust="fdr_bh", p_value=1.0, lfc=0, confint=0.95):
        feature_names = self.gene_name_list
        self.results["logFC"] = pd.Series(self.beta[:, 0], index=feature_names)

        # confidence intervals for LogFC
        if confint:
            alpha = (1.0 + confint) / 2
            margin_error = np.sqrt(self.results["s2_post"]) * self.std_unscaled[:, 0] * t.ppf(alpha,
                                                                                              df=self.df_total)
            self.results["CI.L"] = self.results["logFC"] - margin_error
            self.results["CI.R"] = self.results["logFC"] + margin_error
        # adjusting p-value for multiple testing
        if_passed, adj_pval, alphacSidak, alphacBonf = multipletests(self.results["p_value"][:, 0], alpha=p_value,
                                                                     method=adjust,
                                                                     is_sorted=False, returnsorted=False)
        self.results["adj.P.Val"] = pd.Series(adj_pval, index=feature_names)
        self.results["P.Value"] = pd.Series(self.results["p_value"][:, 0], index=feature_names)
        # make table
        self.table = copy(self.results)
        # remove 'df_prior', 's2_prior', 's2_post', 'df_total','var_prior'
        for key in ['df_prior', 's2_prior', 's2_post', 'var_prior', "p_value"]:
            del self.table[key]
        self.table["t"] = pd.Series(self.table["t"][:, 0], index=feature_names)
        self.table["lods"] = pd.Series(self.table["lods"][:, 0], index=feature_names)
        self.table = pd.DataFrame.from_dict(self.table)

    def e_bayes(self):
        covariate = False  # Amean for limma-trend
        robust = False  #
        winsor_tail_p = (0.05, 0.1)  # needed for fitFDistRobustly()

        self.moderatedT(covariate=covariate, robust=robust, winsor_tail_p=winsor_tail_p)
        self.results["AveExpr"] = self.mean_count

        self.b_stat(std_coef_lim=np.array([0.1, 4]), proportion=0.01)

        self.top_table_t(adjust="fdr_bh", p_value=1.0, lfc=0, confint=0.95)
        self.table = self.table.sort_values(by="P.Value")

    def ebayes_step(self):
        print("making contrasts ...")
        self.make_contrasts(contrast_list=[([self.contrast1_list[0]], [self.contrast2_list[0]])])
        print("contrast matrix:")
        print(self.contrast_matrix)
        print("Fitting contrasts ...")
        self.fit_contrasts()

        print("empirical Bayes ...")
        self.e_bayes()
        print("Done!")
        
        print("Table:")
        print(self.table)
        print('plotting ...')
        self.volcano_plot()
        print('done!')
        
    def prepare_results(self):
        """ Prepare result files for Flimma project """

        try:
            project_result_dir = self.create_result_dir()
            flimma_result_file = f'{project_result_dir}/flimma-result.csv'
            print("Flimma output:",flimma_result_file)
            # save the result file
            self.table.to_csv(flimma_result_file,sep="\t")

        except Exception as io_error:
            logger.error(f"{bcolors.FAIL}Result file write error: {io_error}")
            self.project_failed()

        # ##############  Flimma specific aggregation code

    def volcano_plot(self):
        try:
            table = self.table
            project_result_dir = self.create_result_dir()
            table["gene_names"] = table.index.values
            gnames_to_plot = tuple(table.head(20).index.values)
            visuz.gene_exp.volcano(df=table, lfc='logFC',
                                   pv='adj.P.Val', lfc_thr=(1.0,1.0),
                                   pv_thr=(0.05,0.05), sign_line=True,
                                   genenames=gnames_to_plot, geneid="gene_names", gstyle=2, gfont=8,
                                   show=False, plotlegend=True, legendpos='upper center',
                                   figname=f"{project_result_dir}/p{self.project_id}_volcano", figtype="png",
                                   color=("#E10600FF", "grey", "#00239CFF"), dim=(10, 5))

            # from fed_gwas.models import ResultPlot
            # rp = ResultPlot()
            # rp.project_id = self.project_id
            # rp.file.name = f'p{self.project_id}_volcano.png'
            # rp.save()
        except Exception as exp:
            print(exp)
            visuz.gene_exp.volcano(df=table, lfc='logFC',
                                   pv='adj.P.Val', lfc_thr=(1.0,1.0),
                                   pv_thr=(0.05,0.05), sign_line=True,
                                   genenames=gnames_to_plot, geneid="gene_names", gstyle=2, gfont=8,
                                   show=False, plotlegend=True, legendpos='upper center',
                                   figname=f"{project_result_dir}/p{self.project_id}_volcano", figtype="png",
                                   color=("#E10600FF", "grey", "#00239CFF"), dim=(10, 5))


    def aggregate(self):
        """ perform Flimma project specific aggregations """

        # The following four lines MUST always be called before the aggregation starts
        super().pre_aggregate()
        if self.status != ProjectStatus.AGGREGATING:  # if project failed or aborted, skip aggregation
            super().post_aggregate()
            return

        logger.info(f'Project {self.project_id}: ############## aggregate ####### ')
        logger.info(f'Project {self.project_id}: #### step {self.step}')

        if self.step == HyFedProjectStep.INIT:  # The first step name MUST always be HyFedProjectStep.INIT
            self.init_step()  # global_features, global_cohorts --> clients

        elif self.step == FlimmaProjectStep.PREPARE_INPUTS:
            self.prepare_inputs_step()  # collect features and cohort names; shared_features,cohort effects --> clients

        elif self.step == FlimmaProjectStep.CPM_CUTOFF:
            self.compute_cpm_cutoff_step()  # compute CPM cutoff and --> clients
        elif self.step == FlimmaProjectStep.LIB_SIZES:
            self.get_lib_sizes_step()
        elif self.step == FlimmaProjectStep.APPLY_CPM_CUTOFF:
            self.apply_cpm_cutoff_step()  # compute CPM cutoff and --> clients
        elif self.step == FlimmaProjectStep.COMPUTE_NORM_FACTORS:
            self.compute_norm_factors_step()
        elif self.step == FlimmaProjectStep.LINEAR_REGRESSION:
            self.linear_regression_step()
            self.set_step(FlimmaProjectStep.SSE)
        elif self.step == FlimmaProjectStep.SSE:
            self.sse_step()
            # nothing to broadcast in this step!
            self.set_step(FlimmaProjectStep.MEAN_LOG_COUNTS)
        elif self.step == FlimmaProjectStep.MEAN_LOG_COUNTS:
            self.mean_log_count_step(weighted=False)
            self.global_parameters[FlimmaGlobalParameter.LOWESS] = self.py_lowess
            self.set_step(FlimmaProjectStep.WEIGHTS)
        elif self.step == FlimmaProjectStep.WEIGHTS:
            # nothing to do here!
            self.set_step(FlimmaProjectStep.LINEAR_REGRESSION_WITH_WEIGHTS)
        elif self.step == FlimmaProjectStep.LINEAR_REGRESSION_WITH_WEIGHTS:
            self.linear_regression_step()
            self.set_step(FlimmaProjectStep.SSE_WITH_WEIGHTS)
        elif self.step == FlimmaProjectStep.SSE_WITH_WEIGHTS:
            self.sse_step()
            # nothing to broadcast in this step!
            self.set_step(FlimmaProjectStep.MEAN_LOG_COUNTS_WITH_WEIGHTS)
        elif self.step == FlimmaProjectStep.MEAN_LOG_COUNTS_WITH_WEIGHTS:
            self.mean_log_count_step(weighted=True)
            # nothing to broadcast in this step!
            self.set_step(FlimmaProjectStep.EBAYES)
            
        elif self.step == FlimmaProjectStep.EBAYES:
            self.ebayes_step()
            self.prepare_results()
            self.set_step(HyFedProjectStep.RESULT)
            
        elif self.step == HyFedProjectStep.RESULT:
            super().result_step()

        # The following line MUST be the last function call in the aggregate function
        super().post_aggregate()

    def check_attributes(self, print_values=False):
        pass
        #print(f"{bcolors.WARNING} ######## FlimmaServerProject Attributes #######")
        #check(self.__dict__.items(), print_values)

    def check_global_parameters(self, print_values=True):
        print(f"{bcolors.WARNING} ######## FlimmaGlobalParameter Attributes   #######")
        temp = {}
        for attr, val in FlimmaGlobalParameter.__dict__.items():
            if val in self.global_parameters:
                temp[attr] = self.global_parameters[val]

        check(temp.items(), print_values)


def check(params, print_values):
    for attr, value in params:
        if not attr.startswith('__'):
            t = type(value)
            if isinstance(value, list):
                v = np.array(value)
                dim = v.ndim
                shape = v.shape
                print(f"{bcolors.WARNING}Attribute: {attr},"
                      f" {bcolors.TYPE} type: {t},"
                      f" {bcolors.DIM} dim: {dim},"
                      f" {bcolors.SHAPE}shape: {shape}")
            elif isinstance(value, np.ndarray):
                dim = value.ndim
                shape = value.shape
                print(f"{bcolors.WARNING}Attribute: {attr},"
                      f" {bcolors.TYPE} type: {t},"
                      f" {bcolors.DIM} dim: {dim},"
                      f" {bcolors.SHAPE}shape: {shape}")
            else:
                print(f"{bcolors.WARNING}Attribute: {attr},"
                      f" {bcolors.TYPE} Value: {value}")
            if print_values:
                if isinstance(value, (list, dict, np.ndarray)):
                    print(f"{value[:3]}, Length:{len(value)}")
                else:
                    print(value)


class bcolors:
    SHAPE = '\033[95m'
    TYPE = '\033[94m'
    DIM = '\033[96m'
    VALUE = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
