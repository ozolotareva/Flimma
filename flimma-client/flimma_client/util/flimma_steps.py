"""
    Flimma specific project steps

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


class FlimmaProjectStep:
    PREPARE_INPUTS = "Prepare-Inputs" # exclude not shared genes, ad cohort effects to designs
    CPM_CUTOFF = "Cpm-Cutoff" # precompute filterByExprs() parameters
    LIB_SIZES = "Lib_Sizes"
    APPLY_CPM_CUTOFF = "Apply_Cpm_Cutoff"
    COMPUTE_NORM_FACTORS = "Compute-Norm-Factors" # apply filterByExprs() and compute nomalization factors
    LINEAR_REGRESSION = "Linear-Regression"
    SSE = "Sse"
    MEAN_LOG_COUNTS = "Mean-Log-Counts"
    WEIGHTS = "Weights"
    LINEAR_REGRESSION_WITH_WEIGHTS = "Linear-Regression-With-Weights"
    SSE_WITH_WEIGHTS = "Sse-With-Weights"
    MEAN_LOG_COUNTS_WITH_WEIGHTS = "Mean-Log-Counts-With-Weights"
    EBAYES = "Ebayes"