"""
    Flimma specific server and client parameters

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


class FlimmaProjectParameter:
    NORMALIZATION = "normalization"
    MIN_COUNT = "min_count"
    MIN_TOTAL_COUNT = "min_total_count"
    GROUP1 = "group1"
    GROUP2 = "group2"
    CONFOUNDERS = "confounders"


class FlimmaLocalParameter:
    """ Name of the local (client -> server) parameters """

    SAMPLE_COUNT = "local_sample_count"  # local sample count
    FEATURES = "local_features"  # list of shared features e.g. genes
    COHORT_NAME = "cohort_name"

    # parameters for CPM cutoff
    N_SAMPLES_PER_GRPOUP = "local_n_samples_per_group"  # how many samples in group1 and group2
    TOTAL_COUNT_PER_FEATURE = "local_total_count"
    LIB_SIZES = "local_lib_sizes"

    CPM_CUTOFF_SAMPLE_COUNT = "cpm_cutoff_sample_count"

    UPPER_QUARTILE = "upper_quartile"
    UPDATED_LIB_SIZES = "updated_lib_sizes"

    # linear regression parameters
    XT_X_MATRIX = "xt_x_matrix"
    XT_Y_MATRIX = "xt_y_matrix"

    # SSE parameters
    SSE = "local_sse"
    COVARIANCE_COEFFICIENT = "covariance_coefficient"
    # SAMPLE_COUNT will be reused here

    # mean parameters
    LOG_COUNT = "log_count"
    LOG_COUNT_CONVERSION = "log_count_conversion"

    BETA = "local_beta"


class FlimmaGlobalParameter:
    TOTAL_SAMPLE_COUNT = "global_sample_count"  # sum of local sample counts
    FEATURES = "global_features"  # sorted list of shared features e.g. genes
    COHORT_EFFECTS = "cohort_effects"  # all but one cohort names
    CPM_CUTOFF = "cpm_cutoff"
    GENES_NAME_LIST = "gene_name_list"
    F = "f"
    BETA = "beta"
    LOWESS = "lowess"
