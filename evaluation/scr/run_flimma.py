import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from scipy.interpolate import interp1d
from scipy import linalg
from scipy.special import digamma,polygamma
from scipy.stats import t
import sys,os

#from time import time
np.set_printoptions(precision=22)

from client import Client
from server import Server

import argparse

parser = argparse.ArgumentParser(description='flimma -- federated version of limma123 workflow.')
parser.add_argument('-i', metavar='in_dir/', type=str, nargs=1,
                    help='Folder with input *.counts.tsv and *.design.tsv files.')
parser.add_argument('-o', metavar='out_dir/basename', type=str, nargs=1,
                    help='Output file prefix.')
parser.add_argument('-c','--clients', metavar='cli1 cli2', type=str, nargs='+',
                    help='Prefixes of files indicating client/cohort names. For example for "-c cli1 cli2" there must be cli1.counts.tsv, cli1.design.tsv, and cli2.counts.tsv, cli2.design.tsv files in the input folder.')

parser.add_argument('-vars', metavar='groupA groupB', type=str, nargs=2,
                    help='Names of two sample groups.')
parser.add_argument('-covars', metavar='age stage', type=str, nargs='*',
                    help='Names of covariates to be considered.')

parser.add_argument('--add_cohort_effect',action='store_true',
                    help='whether to add cohort-specific effects to the model')

parser.add_argument('-norm', metavar='UQ', type=str, nargs=1,default=["UQ"],
                    help='Normalization method: UQ (default) or TMM.')


args = parser.parse_args()
print(args)
in_dir = args.i[0]
out_dir_basename = args.o[0]
cohorts  = args.clients
normalization_method = args.norm[0]
add_cohort_effect = args.add_cohort_effect
vars = args.vars
group1 = vars[0]
group2 = vars[1]

if not normalization_method in ["UQ","TMM"]:
    print("Normalization method %s is not recognized. Choose 'UQ' or 'TMM'."%normalization_method,file=sys.stderr)

    
server = Server(args.vars,args.covars)
store_clients = {}

for c in cohorts:
    expression_file_path = in_dir+"/"+c+".counts.tsv" # matrix of raw counts 
    annotation_file_path = in_dir+"/"+c+".design.tsv" # design matrix
    client = Client(c, expression_file_path,annotation_file_path)
    store_clients[client.cohort_name] = client
    
    # join client
    server.join_client(client.cohort_name,client.gene_names,client.n_samples,client.condition_names)
    
print("Client names:",server.client_names)
print("Samples per client:",server.n_samples_per_cli)
print("Variables:",group1,"vs", group2)
print("Confounders:",server.confounders)
print("Shared gene names:", len(server.global_genes))
N = np.sum(server.n_samples_per_cli) # total number of samples
print("Samples in total:",N)

##### validating inputs #####
for c in cohorts:
    client = store_clients[c]
    conditions = server.variables+server.confounders
    client.validate_inputs(server.global_genes,conditions)
    
    # add cohort effect columns to each design matrix
    if add_cohort_effect:
        client.add_cohort_effects_to_design(server.client_names)

# add cohort columns to the list of confounders on the server side
if add_cohort_effect:
    server.confounders = server.confounders+server.client_names[:-1]
else:
    print("possible cohort effects are not modelled")
    server.confounders = server.confounders
print("Variables in the model:", server.confounders)
####################### filterByExprs: Removing lowly expressed genes ########################

# filterByExprs parameters
min_count=10
min_total_count=15
large_n=10
min_prop=0.7
tol = 1e-14

# collect number of samples per group, lib.sizes, and total read counts per gene
n_samples_in_groups = pd.Series(data=np.zeros(len(server.variables)),index=server.variables)
lib_sizes = []
total_counts_per_gene = pd.Series(data=np.zeros(len(server.global_genes)),index=server.global_genes)
for c in store_clients.keys():
    client = store_clients[c]
    n_samples_in_groups+= client.count_samples_in_groups(server.variables)
    total_counts_per_gene += client.get_total_counts_per_gene()
    lib_sizes.append(client.get_libsizes())
lib_sizes  = np.concatenate(lib_sizes)  
print(n_samples_in_groups)

# define min allowed number of samples
min_n_samples = np.min([x for x in n_samples_in_groups if x >0])
if min_n_samples  > large_n:
    min_n_samples = large_n + ( min_n_samples-large_n)*min_prop
print("Min. sample size:",min_n_samples)

# Total count cutoff 
# keep genes if total count is not less than (min_total_count - tol)
keep_total_count = total_counts_per_gene[total_counts_per_gene >= (min_total_count - tol)].index.values
print("Genes passed total count cutoff:",len(keep_total_count))# CPM cutoff
median_lib_size  = np.median(lib_sizes)
CPM_cutoff = min_count/median_lib_size*1e6
print("median lib.size:",median_lib_size,"\nCPM_cutoff:",CPM_cutoff)

# count how many samples pass CPM cutoff
n_samples_passing_CPMcutoff = pd.Series(data=np.zeros(len(server.global_genes)),index=server.global_genes)
for c in store_clients.keys():
    client = store_clients[c]
    n_samples_passing_CPMcutoff += client.count_samples_passing_CPMcutoff(CPM_cutoff)

#keep genes where more than 'min_sample_size' samples passed CPM cutoff
keep_CPM = n_samples_passing_CPMcutoff[n_samples_passing_CPMcutoff>min_n_samples-tol].index.values
print("Genes passed CPM cutoff:",len(keep_CPM))

# keep genes passed both filters
keep_genes = sorted(list(set(keep_CPM).intersection(set(keep_total_count))))

print("Genes passed filterByExprs",len(keep_genes))

for c in store_clients.keys():
    client = store_clients[c]
    client.apply_filter(keep_genes)

# update global genes at the server side
server.global_genes = keep_genes

####################### Normalization ########################
f = [] # unscaled norm.factors 
if normalization_method == "UQ":
    print("Upperquartile Normalization...")
    for c in store_clients.keys():
        client = store_clients[c]
        
        # library sizes and upper quartiles
        client.compute_lib_sizes_and_quartiles()
        f.append(client.unscaled_f)

    
elif normalization_method == "TMM":
    print("TMM Normalization...")
    # compute global average profile to use it as reference 
    avg_profiles = []
    for c in store_clients.keys():
        client = store_clients[c]
        avg_profiles.append(client.get_avg_profile())
    ref = pd.concat(avg_profiles,axis=1).mean(axis=1) # use averaged profile as a reference column
    ref = ref.apply(int)
    
    # compute normalization factors given reference 
    for c in store_clients.keys():
        client = store_clients[c]
        client.compute_TMM_factors(ref)
        f.append(client.unscaled_f)
else:
    print("Normalization method must be 'UQ' or 'TMM'.")
f = np.concatenate(f)
F = server.compute_F(f)
F = np.exp(np.mean(np.log(f)))

# compute normalization factors
for c in store_clients.keys():
    client = store_clients[c]
    client.compute_normalization_factors(F)
    
####################### Voom ########################

### compute XtX, XtY, beta and stdev
XtX_list = []
XtY_list = []
for c in store_clients.keys():
    client = store_clients[c]

    # logCPM
    client.compute_logCPM()
    
    XtX,XtY = client.compute_XtX_XtY(weighted=False)
    XtX_list.append(XtX)
    XtY_list.append(XtY)
server.compute_beta_and_beta_stdev(XtX_list,XtY_list)

### 1) Computes SSE, sigma, and cov. coeficients for clients and aggregates them 
### 2) computes Ameans and log-counts and fits LOWESS
SSE_list = []
cov_coef_list = []
logC_conversion_term = 0
server.Amean = np.zeros(len(server.global_genes))
for c in store_clients.keys():
    client = store_clients[c]
    
    # sum of squared residues
    SSE,cov_coef = client.compute_SSE_and_cov_coef(server.beta, weighted = False)
    SSE_list.append(SSE)
    cov_coef_list.append(cov_coef)
    
    # mean log(counts) per gene
    server.Amean += client.logCPM.sum(axis=1)
    logC_conversion_term += np.sum(np.log2(client.lib_sizes+1))
    
server.aggregate_SSE_and_cov_coef(SSE_list,cov_coef_list)
server.compute_mean_logC(logC_conversion_term)
server.fit_LOWESS_curve()

### 1) converts fitted logCPM to fitted log-counts and
### 2) computes weights 
for c in store_clients.keys():
    client = store_clients[c]
    
    client.calculate_fitted_logcounts(server.beta)
    client.calculate_weights(server.lowess_curve)
####################### lmFit ########################

### Weighted linear regression #######

### 1) applies normalization and 2) computes XtX, XtY, beta and stdev
XtX_list = []
XtY_list = []
for c in store_clients.keys():
    client = store_clients[c]

    XtX,XtY = client.compute_XtX_XtY(weighted=True)
    XtX_list.append(XtX)
    XtY_list.append(XtY)
server.compute_beta_and_beta_stdev(XtX_list,XtY_list)

### 1) Computes SSE, sigma, and cov. coeficients for clients and aggregates them 2) computes Ameans and log-counts and fits LOWESS
SSE_list = []
cov_coef_list = []
logC_conversion_term = 0
server.Amean = np.zeros(len(server.global_genes))
for c in store_clients.keys():
    client = store_clients[c]
    
    # sum of squared residues
    SSE,cov_coef = client.compute_SSE_and_cov_coef(server.beta, weighted = True)
    SSE_list.append(SSE)
    cov_coef_list.append(cov_coef)
    
    # mean log(counts) per gene
    server.Amean += client.logCPM.sum(axis=1)
    logC_conversion_term += np.sum(np.log2(client.lib_sizes+1))
    
server.aggregate_SSE_and_cov_coef(SSE_list,cov_coef_list,weighted = True)
server.compute_mean_logC(logC_conversion_term,weighted = True)
server.fit_LOWESS_curve(weighted=True)

server.plot_voom_results(svg_file=out_dir_basename+".voom_results.svg")
####################### eBayes ########################
# contrasts matrix
contrast_matrix = server.make_contrasts(contrasts=[([group1],[group2])])
### applies contrasts
server.fit_contasts(contrast_matrix.values)


server.eBayes()
####################### output results ########################
# result 

server.table.to_csv(out_dir_basename+".flimma_results.tsv",sep="\t")

