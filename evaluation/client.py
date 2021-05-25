import pandas as pd
import numpy as np
import sys
from scipy import linalg

### voom weights ###

def rpy2_weights(fitted_logcount,lowess_curve):
    '''runs R stats::approxfun for approximation'''
    #t0 = time()
    from rpy2.robjects.packages import importr
    from rpy2.rinterface import ListSexpVector
    from rpy2.robjects import numpy2ri,pandas2ri
    
    R_fitted_logcount = numpy2ri.py2rpy(fitted_logcount)
    R_x = numpy2ri.py2rpy(lowess_curve[:,0])
    R_y = numpy2ri.py2rpy(lowess_curve[:,1])
    
    base = importr('base')
    stats = importr('stats')


    ties = ListSexpVector(["ordered",base.mean])

    R_lo = stats.approxfun(R_x,R_y,rule=2,ties=ties)
    R_Weights  = R_lo(R_fitted_logcount)
    R_Weights = np.array(R_Weights)**-4
    R_Weights = R_Weights.reshape(fitted_logcount.shape[1], fitted_logcount.shape[0]).T
    
    #print(round(time()-t0,2))
    return R_Weights

def weights(fitted_logcount,lowess_curve):
    # lo: returns y = lo(x) given LOWESS curve
    #t0 = time()
    from scipy.interpolate import interp1d
    lo = interp1d(lowess_curve[:,0],lowess_curve[:,1],kind="nearest",fill_value="extrapolate")
    Weights  = lo(fitted_logcount)**-4
    #print(round(time()-t0,2))
    return Weights

class Client:
    def __init__(self, cohort_name, expression_file_path, design_file_path):
        self.cohort_name = cohort_name
        self.gene_names = None
        self.sample_names = None
        self.logCPM = None
        self.CPM = None
        self.lib_sizes = None
        self.upper_quartiles = None
        self.unscaled_f  = None # UQ or TMM norm.factors before division by F
        self.norm_factors = None 
        self.open_dataset(expression_file_path, design_file_path)
        
        self.XtX = None
        self.XtY = None
        self.weights = None # don't forget sqrt(weights)
        self.SSE = None
        self.cov_coef = None
        self.fitted_logcounts = None
        self.mu = None
        
    def open_dataset(self, expression_file_path, design_file_path):
        '''Reads expression and design matrices and ensures that sample names are the same.
        Only samples presenting in both matrices are kept.'''
        
        self.raw_counts = pd.read_csv(expression_file_path,sep ="\t",index_col =0 ) # matrix of raw counts 
        self.gene_names = list(self.raw_counts.index.values) 
        self.sample_names = list(self.raw_counts.columns.values) 
        self.n_samples = self.raw_counts.shape[1]
        
        
        self.design = pd.read_csv(design_file_path,sep ="\t",index_col =0 ) # sample annotation table
        self.condition_names = list(self.design.columns.values)
        
        # ensure that design row and expression column names are the same and in the same order
        design_rows = self.design.index.values
        exprs_cols = self.raw_counts.columns.values
        if not np.any(design_rows == exprs_cols):
            design_rows = set(design_rows)
            exprs_cols = set(exprs_cols)
            if not design_rows == exprs_cols:
                keep_samples = sorted(design_rows.intersection(exprs_cols))
                print("Row names of design matrix does not match column names in expression matrix.",file= sys.stderr)
                not_found_in_exprs = exprs_cols.difference(keep_samples)
                not_found_in_design = design_rows.difference(keep_samples)
                if len(not_found_in_exprs)>0:
                    print("%s samples not found in the expression matrix:"%len(not_found_in_exprs),not_found_in_exprs,file= sys.stderr)
                if len(not_found_in_design)>0:
                    print("%s samples not found in the design matrix:"%len(not_found_in_design),not_found_in_design,file= sys.stderr)
            # reorder genes (and drop extra if any)
            self.design = self.design.loc[keep_samples,:]
            self.raw_counts = self.raw_counts.loc[:,keep_samples]
        self.norm_factors = np.ones(self.raw_counts.shape[1])
        self.lib_sizes = self.raw_counts.sum(axis=0)
    
    ### Client: returns gene_names, n_samples, and condition_names are sent to Server
    ### Server: returns global_gene_names, conditions (defined by the user: contrasted variables + confounder variables )
    
    def validate_inputs(self, global_gene_names,conditions):
        '''Checks if gene_names match global gene names and remove unnecessary genes.'''
        # ensure that gene names are the same and are in the same order
        if self.gene_names != global_gene_names:
            self_genes= set(self.gene_names)
            global_genes = set(global_gene_names)
            if not self_genes == global_genes: # drop extra genes
                extra_genes = self_genes.difference(global_genes)
                if len (extra_genes)>0:
                    print("Client %s:\t %s genes absent in other datasets are dropped:"%(len(extra_genes),self.cohort_name), file = sys.stderr)
            # reorder genes (and drop extra if any)
            self.raw_counts = self.raw_counts.loc[global_gene_names,:]
            self.gene_names = global_gene_names
            # ensure that annotation table contains the same columns as conditions
        if self.condition_names != conditions:
            cli_consitions  = set(self.condition_names)
            glob_conditions = set(conditions)
            missing_conditions = glob_conditions.difference(cli_consitions)
            if len(missing_conditions)>0:
                print("Some conditions are missing in the design matrix:",missing_conditions, file = sys.stderr)
                exit(1)
            extra_conditions = cli_consitions.difference(glob_conditions)
            
            if len(extra_conditions)>0:
                print("%s conditions are excluded from the design matrix:"%len(extra_conditions),extra_conditions,file=sys.stderr)
            self.design = self.design.loc[:,conditions]
    
    ### Server: returns the list of all cohort ids
    def add_cohort_effects_to_design(self,cohorts):
        # add covariates to model cohort effects
        for cohort in cohorts[:-1]: # add 1 column less than the number of cohorts
            if self.cohort_name == cohort:
                self.design[cohort]=1
            else:
                self.design[cohort]=0
        self.condition_names = sorted(list(self.design.columns.values))
                
    ######### Filtering by median ########
    def shuffle_expressions(self):
        '''Reorders expression values within each gene.'''
        shuffled_counts = np.zeros(self.raw_counts.shape)
        for i in range(0,self.raw_counts.shape[0]):
            shuffled_counts[i,:] = np.sort(self.raw_counts.values[i,:])
        shuffled_counts =  pd.DataFrame(data=shuffled_counts,index=self.gene_names) # shuffled matrices are sent to the server
        return shuffled_counts
    
    def apply_filter(self, keep_genes):
        self.raw_counts = self.raw_counts.loc[keep_genes,:]
        self.gene_names = keep_genes
        
    ######### Normalization factors ######
    def get_nonzero_genes(self):
        row_max = self.raw_counts.max(axis=1)
        non_zero_genes = list(row_max[row_max>0].index.values)
        return non_zero_genes
    
    def compute_sum_log_f(self):
        lib_sizes = self.raw_counts.sum().values
        # replace 0 with nan
        #nan_raw_counts = self.raw_counts
        #nan_raw_counts  = nan_raw_counts.replace(0, np.nan).values
        # upper quartiles 
        #upper_quartiles = np.nanquantile(self.raw_counts.values,0.75,axis=0)
        upper_quartiles = self.raw_counts.quantile(0.75).values
        self.unscaled_f = upper_quartiles/lib_sizes
        sum_log_f = np.sum(np.log(self.unscaled_f))
        return sum_log_f
    
    def compute_lib_sizes_and_quartiles(self):
        self.lib_sizes = self.raw_counts.sum().values
        self.upper_quartiles = self.raw_counts.quantile(0.75).values
        self.unscaled_f = self.upper_quartiles/self.lib_sizes

    def compute_normalization_factors(self,F):
        self.norm_factors = self.unscaled_f/F
        print(self.norm_factors)
        
    ######### Log CPM #############

    # calculates Y from raw counts 
    def compute_logCPM(self, add=0.5):
        '''Calculates normalized log2(CPM) from raw counts.'''
        self.lib_sizes = self.raw_counts.sum(axis=0) # update lib. sizes
        self.logCPM = self.raw_counts.applymap(lambda x :x+add)/(self.lib_sizes*self.norm_factors+1)*10**6
        self.logCPM = self.logCPM.applymap(lambda x: np.log2(x))
        
    """def edgeR_logCPM(self,mean_norm_libsize, prior_count=2):
        '''Calculates normalized log2(CPM) from raw counts.'''
        # in edgeR lcpm is different: lcpm = log2(cpm+x),
        # where x ~ prior.count*1e6/(mean(exprs$samples$lib.size*exprs$samples$norm.factors)+1)
        self.lib_sizes = self.raw_counts.sum(axis=0) # update lib. sizes
        self.logCPM = self.raw_counts/(self.lib_sizes*self.norm_factors+1)*10**6
        add = prior_count*1e6/(mean_norm_libsize)
        self.logCPM = self.logCPM.applymap(lambda x: np.log2(x + add))
    """
    
    ####### linear regression #########
    def compute_XtX_XtY(self,weighted = False):
        X = self.design.values
        Y = self.logCPM.values # Y - logCPM (samples x genes)
        n = Y.shape[0] # genes
        k = self.design.shape[1] # conditions
        self.XtX = np.zeros((n,k,k))
        self.XtY = np.zeros((n,k))
        self.mu = np.zeros(Y.shape)
    
        if weighted:
            weighted = True
            W = np.sqrt(self.weights)
            Y = np.multiply(Y,W)  # algebraic multiplications by W

        # linear models for each row
        for i in range(0,n): # 
            y = Y[i,:]
            if weighted:
                Xw = np.multiply(X,W[i,:].reshape(-1, 1)) # algebraic multiplications by W
                self.XtX[i,:,:] = Xw.T @ Xw 
                self.XtY[i,:] = Xw.T @ y  
            else:
                self.XtX[i,:,:] = X.T @ X 
                self.XtY[i,:] = X.T @ y 
        return self.XtX, self.XtY
    
    def compute_SSE_and_cov_coef(self,beta, weighted = False):
        X = self.design.values
        Y = self.logCPM.values 
        n = Y.shape[0]
        self.SSE = np.zeros(n)
        if weighted:
            W = np.sqrt(self.weights)
            Y = np.multiply(Y,W)
        for i in range(0,n): # 
            y = Y[i,:]
            if weighted:
                Xw = np.multiply(X,W[i,:].reshape(-1, 1))
                self.mu[i,] =  Xw @ beta[i,:] # fitted logCPM 
            else:
                self.mu[i,] =  X @ beta[i,:] # fitted logCPM 

            self.SSE[i] = np.sum((y - self.mu[i,])**2) # local SSE
        #print("mu:",self.mu.shape)
        Q,R = np.linalg.qr(X)
        self.cov_coef = R.T @ R
        self.cov_coef = X.T @ X
        return self.SSE, self.cov_coef

    
    ### Fitted Log-counts #####
    def calculate_fitted_logcounts(self,beta):
        '''Converts fitted logCPM back to fitted log-counts.'''
        fitted_counts = (2**self.mu.T)*10**-6 # fitted logCPM -> fitted CPM -> fitted counts/norm_lib_size
        norm_lib_sizes = self.lib_sizes*self.norm_factors+1
        fitted_counts = np.multiply(fitted_counts, norm_lib_sizes.values.reshape(-1, 1)).T
        self.fitted_logcounts = np.log2(fitted_counts)
    
    def calculate_weights(self,lowess_curve):
        #self.weights = rpy2_weights(self.fitted_logcounts,lowess_curve)
        self.weights = weights(self.fitted_logcounts,lowess_curve)

##### for Federated median #########
    def get_local_median_lb_ub(self):
        '''lower and upper bound median of raw counts'''
        lb = self.raw_counts.min(axis=1)
        ub = self.raw_counts.max(axis=1)
        sigma = self.raw_counts.std(axis=1)
        return lb-sigma, ub+sigma
    
    def get_n_le_gt(self,m):
        # how many values are below m (row-wise)
        df = self.raw_counts.loc[m.index.values, :]
        le = df[df.apply(lambda col: col <= m)].T.count()
        gt = df[df.apply(lambda col: col > m)].T.count()
        le_gt = pd.concat([le,gt],axis=1)
        le_gt.columns = ["le","gt"]
        return le_gt
    
    def find_max_of_lesser(self,m):
        df = self.raw_counts.loc[m.index.values, :]
        return df[df.apply(lambda col: col <= m)].max(axis=1)
         
    def find_min_of_greater(self,m):
        df = self.raw_counts.loc[m.index.values, :]
        return df[df.apply(lambda col: col > m)].min(axis=1)

    
### TMM normalization
    def get_libsizes(self):
        return self.lib_sizes
    '''
    def get_avg_profile(self):
        return self.raw_counts.mean(axis=1)
    
    def calc_TMM_factor(self,x,ref,logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10):
        x = 1.0*x # profile to be transformed 
        ref = 1.0*ref # reference profile
        # lib. sizes
        l_x = np.sum(x)
        l_ref = np.sum(ref)

        logR = (x/l_x)/(ref/l_ref) # log ratio of expression, accounting for library size
        absE = (x/l_x) * (ref/l_ref) # absolute expression
        v = (l_x-x)/l_x/x + (l_ref-ref)/l_ref/ref # estimated asymptotic variance

        df = pd.concat([logR,absE,v],axis=1)
        df.columns = ["logR","absE","v"]
        df["logR"] = df["logR"].apply(np.log2)
        df["absE"] = df["absE"].apply(np.log2)/2
        # remove all genes with nans and infs 
        df = df.dropna()
        df = df.loc[~np.isinf(df).any(axis=1),:]
        # remove all genes where absE > Acutoff
        df = df.loc[df["absE"] > Acutoff]

        if np.max(np.abs(df["logR"].values)) < 1e-6:
            return 1

        # taken from the original mean() function
        n = df.shape[0]
        loL = np.floor(n * logratioTrim) + 1
        hiL = n + 1.0 - loL
        loS = np.floor(n * sumTrim) + 1
        hiS = n + 1 - loS
        #print(loL,hiL,loS,hiS)

        # non-integer values when there are a lot of ties
        df_ranks = df[["logR","absE"]].rank() 
        df_ranks = df_ranks.loc[df_ranks["logR"]>=loL]
        df_ranks = df_ranks.loc[df_ranks["logR"]<=hiL]
        df_ranks = df_ranks.loc[df_ranks["absE"]>=loS]
        df_ranks = df_ranks.loc[df_ranks["absE"]<=hiS]
        df = df.loc[df_ranks.index,:]

        #keep = (df["logR"].rank()>=loL & np.rank(df["logR"])<=hiL) & (np.rank(df["absE"])>=loS & rank(absE)<=hiS)

        if doWeighting:
            f = np.sum(logR/v) / np.sum(1.0/v)
        else:
            f = np.mean(logR)

        return 2**f
    
    def compute_TMM_factors(self, ref):
        norm_factors = []
        for s in self.raw_counts.columns.values:
            x = self.raw_counts[s]
            f = self.calc_TMM_factor(x,ref,logratioTrim=0.3, sumTrim=0.05, doWeighting=True, Acutoff=-1e10)
            norm_factors.append(f)
        self.unscaled_f  =  np.array(norm_factors)
    '''
##### filterByExprs ####
    def count_samples_in_groups(self,variables):
        design = self.design.loc[:,variables]
        return design.sum(axis=0)
    
    def get_total_counts_per_gene(self):
        return self.raw_counts.sum(axis=1)
    
    def count_samples_passing_CPMcutoff(self,CPM_cutoff):
        self.compute_CPM()
        #keep genes where at least 'min_sample_size' samples pass CPM cutoff
        n_samples_passing_CPMcutoff = self.CPM[self.CPM>=CPM_cutoff].count(axis=1)
        return n_samples_passing_CPMcutoff # per gene
    
        # calculates Y from raw counts 
    def compute_CPM(self):
        '''Calculates normalized CPM from raw counts.'''
        #self.CPM = self.raw_counts.applymap(lambda x:x)
        self.lib_sizes = self.raw_counts.sum(axis=0)
        self.CPM = self.raw_counts/(self.lib_sizes*self.norm_factors+1)*10**6