"""
    Client-side Flimma project to compute local parameters

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

from hyfed_client.project.hyfed_client_project import HyFedClientProject
from hyfed_client.util.hyfed_steps import HyFedProjectStep

from flimma_client.util.flimma_steps import FlimmaProjectStep
from flimma_client.util.flimma_parameters import FlimmaGlobalParameter, FlimmaLocalParameter
from flimma_client.util.flimma_algorithms import FlimmaAlgorithm

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class FlimmaClientProject(HyFedClientProject):
    """ A class that provides the computation functions to compute local parameters

    Attributes
    ----------
    group1:
    group2:
    confounders:
    flimma_counts_file_path: str
    flimma_design_file_path: str
    local_features: list
        list of genes
    samples: list
        shared samples in both design and counts files
    local_sample_count: int
        the number of shared samples in both input files
    variables: list
        all relevant variables
    design_df: pandas.core.frame.DataFrame
    counts_df: pandas.core.frame.DataFrame
    norm_factors: numpy.array
    upper_quartile: list
    lib_sizes: list
    log_cpm: pandas.core.frame.DataFrame
    xt_x: numpy.array
    xt_y: numpy.array
    mu: numpy.array
    sse: numpy.array
    beta: list

    """

    def __init__(self, username, token, project_id, server_url, compensator_url,
                 algorithm, name, description, coordinator, result_dir, log_dir,
                 flimma_counts_file_path="",  # Flimma specific parameters
                 flimma_design_file_path="",
                 normalization="",
                 min_count=0, min_total_count=0,
                 group1="", group2="", confounders=""):

        super().__init__(username=username, token=token, project_id=project_id, server_url=server_url,
                         compensator_url=compensator_url, algorithm=algorithm, name=name, description=description,
                         coordinator=coordinator, result_dir=result_dir, log_dir=log_dir)

        self.group1 = sorted(
            [label.strip() for label in group1.split(',')])  # assume group1 and 2 may be ,-separated lists
        self.group2 = sorted([label.strip() for label in group2.split(',')])
        if len(confounders) == 0:
            self.confounders = []
        else:
            self.confounders = sorted([confounder.strip() for confounder in confounders.split(',')])

        # Flimma specific dataset related attributes
        self.flimma_counts_file_path = flimma_counts_file_path
        self.flimma_design_file_path = flimma_design_file_path
        self.local_features = []
        self.samples = []
        self.local_sample_count = 0
        self.normalization = normalization
        self.min_count = min_count
        self.min_total_count = min_total_count
        self.counts_df = pd.DataFrame()
        self.design_df = pd.DataFrame()
        self.variables = None
        self.cohort_name = ""
        self.norm_factors = None
        self.lib_sizes = []
        self.upper_quartile = []
        self.log_cpm = pd.DataFrame()
        self.sse = np.array([])
        self.xt_x = np.array([])
        self.xt_y = np.array([])
        self.mu = np.array([])
        self.sse = np.array([])
        self.beta = []
        self.cov_coefficient = None
        self.fitted_log_counts = None
        self.weight = None

    # ########## Flimma step functions
    def init_step(self):
        # OPEN DATASET FILE(S) AND INITIALIZE THEIR CORRESPONDING DATASET ATTRIBUTES
        try:

            # read input files
            counts_df = pd.read_csv(self.flimma_counts_file_path, index_col=0, sep="\t")
            design_df = pd.read_csv(self.flimma_design_file_path, index_col=0, sep="\t")
            self.local_features = sorted(counts_df.index.values)  # features (e.g. genes)
            # ensure that the same samples are in columns of count matrix and on rows of design matrix
            self.samples = sorted(list(set(counts_df.columns.values).intersection(design_df.index.values)))
            self.local_sample_count = len(self.samples)  # the number of samples found in both input files

            # ensure all target and confounder variables are in design matrix columns
            design_cols = set(design_df.columns)
            if len(set(self.group1).intersection(design_cols)) == 0:
                self.log("\tClass labels %s are missing in the design matrix." % ",".join(self.group1))
                self.set_operation_status_failed()
            if len(set(self.group2).intersection(design_cols)) == 0:
                self.log("\t Class labels %s are missing in the design matrix." % ",".join(self.group2))
                self.set_operation_status_failed()
            missing_conf_variables = set(self.confounders).difference(set(design_cols))
            if len(missing_conf_variables) > 0:
                self.log(
                    "\tConfounder variable(s) are missing in the design matrix: %s." % ",".join(missing_conf_variables))
                self.set_operation_status_failed()

            # sort and save inputs
            self.counts_df = counts_df.loc[self.local_features, self.samples]
            self.variables = self.group1 + self.group2 + self.confounders  # list of all relevant variables
            self.design_df = design_df.loc[self.samples, self.variables]
            self.log("\tCount matrix: %s features x %s samples" % self.counts_df.shape)

            # name cohort
            self.cohort_name = "Cohort_" + self.username  # name is unique -- user can join the project only once

            # send sample counts to the server
            self.local_parameters[FlimmaLocalParameter.SAMPLE_COUNT] = self.local_sample_count
            self.norm_factors = np.ones(self.counts_df.shape[1])
        except Exception as io_exception:
            self.log(f'Client: init_step() is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def prepare_inputs_step(self):

        # send the list of features and cohort names to the server
        self.local_parameters[FlimmaLocalParameter.FEATURES] = self.local_features
        self.local_parameters[FlimmaLocalParameter.COHORT_NAME] = self.cohort_name

    def compute_cpm_cutoff_step(self):
        """ Keep only columns with contrast variables and confounders, and rows with shared features.
        Add (n_clients -1 ) variables modeling cohort effects to the design matrix.
        Precompute paramters for CPM cutoff and send them to server."""

        try:
            # receive shared feature names
            shared_features = self.global_parameters[FlimmaGlobalParameter.FEATURES]
            if len(shared_features) != len(self.local_features):
                self.log(f'filter_inputs_step():\t'
                         f'{len(self.local_features) - len(shared_features)}'
                         f' features absent in other datasets are excluded.')
            # keep only shared features in count matrix and update local features
            self.counts_df = self.counts_df.loc[shared_features, :]
            self.local_features = shared_features

            # receive the list of cohort names (i.e. client ids)
            cohort_effects = self.global_parameters[FlimmaGlobalParameter.COHORT_EFFECTS]
            # add variables modeling cohort effects to the design matrix for all but the last cohorts
            for cohort in cohort_effects:  # add n-1 columns for n cohorts
                if self.cohort_name == cohort:
                    self.design_df[cohort] = 1  # add a column of ones for the current cohort
                else:
                    self.design_df[cohort] = 0  # otherwise, add a column of zeroes
            # testprints
            print(cohort_effects)
            print(self.design_df.head(2))

            # compute local parameters for CPM cutoff
            self.variables = self.group1 + self.group2 + self.confounders+cohort_effects
            self.design_df = self.design_df.loc[:, self.variables]
            
            # send to server local parameters for CPM cutoff
            # self.local_parameters[FlimmaLocalParameter.N_SAMPLES_PER_GRPOUP] = self.design_df.sum(axis=0).values
            # self.local_parameters[FlimmaLocalParameter.TOTAL_COUNT_PER_FEATURE] = self.counts_df.sum(axis=1).values
            # self.local_parameters[FlimmaLocalParameter.LIB_SIZES] = self.counts_df.sum(axis=0).values

        except Exception as io_exception:
            self.log(f'Client: compute_cpm_cutoff_step() is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def apply_cpm_cutoff_step(self):
        """Apply CPM Cutoff.
        Precompute parameters for normalization: li"""
        try:
            cpm_cutoff = self.global_parameters[FlimmaGlobalParameter.CPM_CUTOFF]
            # self.log(f"Client: {self.project_id}: cpm_cutoff: {cpm_cutoff}")
            lib_sizes = self.counts_df.sum(axis=0)
            # self.log(f"Client: {self.project_id}: lib_sizes: {lib_sizes}")
            cpm = self.counts_df / (lib_sizes * self.norm_factors + 1) * 10 ** 6
            # self.log(f"cpm: {self.project_id}: {cpm}")
            # cpm_cutoff_sample_count = cpm[cpm >= cpm_cutoff].size
            cpm_cutoff_sample_count = cpm[cpm >= cpm_cutoff].apply(lambda x: sum(x.notnull().values), axis=1).values
            # self.log(f"Client: {self.project_id}: cpm_cutoff_sample_count: {cpm_cutoff_sample_count}")
            self.local_parameters[FlimmaLocalParameter.CPM_CUTOFF_SAMPLE_COUNT] = cpm_cutoff_sample_count

        except Exception as io_exception:
            self.log(f'Client: apply_cpm_cutoff_step() is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def compute_norm_factors_step(self):
        try:
            filtered_genes = self.global_parameters[FlimmaGlobalParameter.GENES_NAME_LIST]
            self.counts_df = self.counts_df.loc[filtered_genes, :]
            self.lib_sizes = self.counts_df.sum().values
            self.upper_quartile = self.counts_df.quantile(0.75).values
            self.local_parameters[FlimmaLocalParameter.UPPER_QUARTILE] = self.upper_quartile
            self.local_parameters[FlimmaLocalParameter.UPDATED_LIB_SIZES] = self.lib_sizes

        except Exception as io_exception:
            self.log(f'Client: compute_norm_factors_step() is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def linear_regression_step(self, weighted):
        try:
            if not weighted:
                f = self.global_parameters[FlimmaGlobalParameter.F]
                self.norm_factors = self.upper_quartile / self.lib_sizes / f
            self.compute_log_cpm()
            self.compute_linear_regression_parameters(weighted)
            self.local_parameters[FlimmaLocalParameter.XT_X_MATRIX] = self.xt_x
            self.local_parameters[FlimmaLocalParameter.XT_Y_MATRIX] = self.xt_y

            self.set_compensator_flag()
        except Exception as io_exception:
            self.log(
                f'Client: linear_regression(weighted={weighted}) is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def compute_log_cpm(self, add=0.5, log2=True):
        self.log_cpm = self.counts_df.applymap(lambda x: x + add)
        self.log_cpm = self.log_cpm / (self.lib_sizes * self.norm_factors + 1) * 10 ** 6
        if log2:
            self.log_cpm = self.log_cpm.applymap(lambda x: np.log2(x))

    def compute_linear_regression_parameters(self, weighted):
        x_matrix = self.design_df.values
        y_matrix = self.log_cpm.values  # Y - logCPM (samples x genes)
        n = y_matrix.shape[0]  # genes
        k = self.design_df.shape[1]  # conditions
        self.xt_x = np.zeros((n, k, k))
        self.xt_y = np.zeros((n, k))
        self.mu = np.zeros(y_matrix.shape)

        if weighted:
            w_square = np.sqrt(self.weight)
            y_matrix = np.multiply(y_matrix, w_square)  # algebraic multiplications by W

        # linear models for each row
        for i in range(0, n):  #
            y = y_matrix[i, :]
            if weighted:
                x_w = np.multiply(x_matrix, w_square[i, :].reshape(-1, 1))  # algebraic multiplications by W
                self.xt_x[i, :, :] = x_w.T @ x_w
                self.xt_y[i, :] = x_w.T @ y
            else:
                self.xt_x[i, :, :] = x_matrix.T @ x_matrix
                self.xt_y[i, :] = x_matrix.T @ y

    def sse_step(self, weighted):
        try:
            self.beta = self.global_parameters[FlimmaGlobalParameter.BETA]
            self.compute_sse_step_parameters(weighted)
            self.local_parameters[FlimmaLocalParameter.SSE] = self.sse
            self.local_parameters[FlimmaLocalParameter.COVARIANCE_COEFFICIENT] = self.cov_coefficient
            # SAMPLE_COUNT (self.local_sample_count) is provided in previous steps for the coordinator
            # it seems unnecessary to send it again!
            self.local_parameters[FlimmaLocalParameter.SAMPLE_COUNT] = self.local_sample_count

            self.set_compensator_flag()

        except Exception as io_exception:
            self.log(f'Client: sse_step(weighted={weighted}) is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def compute_sse_step_parameters(self, weighted):
        x_matrix = self.design_df.values
        y_matrix = self.log_cpm.values
        n = y_matrix.shape[0]
        self.sse = np.zeros(n)
        if weighted:
            w_square = np.sqrt(self.weight)
            y_matrix = np.multiply(y_matrix, w_square)

        for i in range(0, n):  #
            y = y_matrix[i, :]
            if weighted:
                x_w = np.multiply(x_matrix, w_square[i, :].reshape(-1, 1))
                self.mu[i,] = x_w @ self.beta[i, :]  # fitted logCPM
            else:
                self.mu[i,] = x_matrix @ self.beta[i, :]  # fitted logCPM

            self.sse[i] = np.sum((y - self.mu[i,]) ** 2)  # local SSE

        Q, R = np.linalg.qr(x_matrix)
        self.cov_coefficient = R.T @ R

    def mean_log_count_step(self):
        try:
            # Exact procedure will be repeated two times! (weighted = {True, False})
            # Get nothing from the server!
            log_count = self.log_cpm.sum(axis=1).values

            log_count_conversion_term = np.sum(np.log2(self.lib_sizes + 1))

            self.local_parameters[FlimmaLocalParameter.LOG_COUNT] = log_count
            self.local_parameters[FlimmaLocalParameter.LOG_COUNT_CONVERSION] = log_count_conversion_term.item()

            self.set_compensator_flag()
        except Exception as io_exception:
            self.log(f'Client: mean_log_count_step() is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def weight_step(self):
        try:
            py_lowess = self.global_parameters[FlimmaGlobalParameter.LOWESS]

            '''Converts fitted logCPM back to fitted log-counts.'''
            fitted_counts = (2 ** self.mu.T) * 10 ** -6  # fitted logCPM -> fitted CPM -> fitted counts/norm_lib_size
            norm_lib_sizes = self.lib_sizes * self.norm_factors + 1
            fitted_counts = np.multiply(fitted_counts, norm_lib_sizes.reshape(-1, 1)).T
            self.fitted_log_counts = np.log2(fitted_counts)

            lo = interp1d(py_lowess[:, 0], py_lowess[:, 1], kind="nearest", fill_value="extrapolate")
            self.weight = lo(self.fitted_log_counts) ** -4
        except Exception as io_exception:
            self.log(f'Client: weight_step() is failed for project {self.project_id}: {io_exception}')
            self.set_operation_status_failed()

    def compute_local_parameters(self):
        """ Compute the local parameters in each step of the Flimma algorithms """

        try:

            super().pre_compute_local_parameters()  # MUST be called BEFORE step functions

            # ############## Flimma specific local parameter computation steps
            if self.project_step == HyFedProjectStep.INIT:
                # read, filter and sort inputs;  local_sample_count -> server
                self.init_step()
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.PREPARE_INPUTS:
                self.prepare_inputs_step()
            elif self.project_step == FlimmaProjectStep.CPM_CUTOFF:
                self.compute_cpm_cutoff_step()
                self.local_parameters[FlimmaLocalParameter.N_SAMPLES_PER_GRPOUP] = self.design_df.sum(axis=0).values
                self.local_parameters[FlimmaLocalParameter.TOTAL_COUNT_PER_FEATURE] = self.counts_df.sum(axis=1).values
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.LIB_SIZES:
                self.local_parameters[FlimmaLocalParameter.LIB_SIZES] = self.counts_df.sum(axis=0).values
            elif self.project_step == FlimmaProjectStep.APPLY_CPM_CUTOFF:
                self.apply_cpm_cutoff_step()
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.COMPUTE_NORM_FACTORS:
                self.compute_norm_factors_step()
            elif self.project_step == FlimmaProjectStep.LINEAR_REGRESSION:
                self.linear_regression_step(weighted=False)
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.SSE:
                self.sse_step(weighted=False)
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.MEAN_LOG_COUNTS:
                self.mean_log_count_step()
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.WEIGHTS:
                self.weight_step()
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.LINEAR_REGRESSION_WITH_WEIGHTS:
                self.linear_regression_step(weighted=True)
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.SSE_WITH_WEIGHTS:
                self.sse_step(weighted=True)
                self.set_compensator_flag()
            elif self.project_step == FlimmaProjectStep.MEAN_LOG_COUNTS_WITH_WEIGHTS:
                self.mean_log_count_step()
                self.set_compensator_flag()
            elif self.project_step == HyFedProjectStep.RESULT:
                super().result_step()  # the result step downloads the result file as zip (it is algorithm-agnostic)
            elif self.project_step == HyFedProjectStep.FINISHED:
                super().finished_step()  # The operations in the last step of the project is algorithm-agnostic
            self.check_global_parameters()
            self.check_local_parameters()
            super().post_compute_local_parameters()  # # MUST be called AFTER step functions
        except Exception as computation_exception:
            self.log(computation_exception)
            super().post_compute_local_parameters()
            self.set_operation_status_failed()

    def check_attributes(self):
        print(f"{bcolors.WARNING} ######## FlimmaClientProject Attributes  **  {self.project_step} **   #######")
        check(self.__dict__.items())

    def check_local_parameters(self, print_values=False):
        print(f"{bcolors.WARNING} ######## FlimmaLocalParameter Attributes  **  {self.project_step} **   #######")
        temp = {}
        for attr, val in FlimmaLocalParameter.__dict__.items():
            if val in self.local_parameters:
                temp[attr] = self.local_parameters[val]
        check(temp.items(), print_values)

    def check_global_parameters(self, print_values=False):
        print(f"{bcolors.WARNING} ######## FlimmaGlobalParameter Attributes **  {self.project_step} **    #######")
        temp = {}
        for attr, val in FlimmaGlobalParameter.__dict__.items():
            if val in self.global_parameters:
                temp[attr] = self.global_parameters[val]

        check(temp.items(), print_values)


def check(params, print_values=False):
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
                      f" {bcolors.SHAPE}shape: {shape},"
                      f" {bcolors.VALUE} Value: {value[:3]}...")
            elif isinstance(value, np.ndarray):
                dim = value.ndim
                shape = value.shape
                print(f"{bcolors.WARNING}Attribute: {attr},"
                      f" {bcolors.TYPE} type: {t},"
                      f" {bcolors.DIM} dim: {dim},"
                      f" {bcolors.SHAPE}shape: {shape},"
                      f" {bcolors.VALUE} Value: {value[:3]}...")
            else:
                print(f"{bcolors.WARNING}Attribute: {attr},"
                      f" {bcolors.VALUE} Value: {value}")
            if print_values:
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
