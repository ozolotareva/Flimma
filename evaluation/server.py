import pandas as pd
import numpy as np
import sys
from copy import copy
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from scipy.interpolate import interp1d
from scipy import linalg
from scipy.special import digamma,polygamma
from scipy.stats import t

import matplotlib.pyplot as plt

def cov2cor(cov_coef):
    cor  = np.diag(cov_coef)**-0.5 * cov_coef
    cor = cor.T * np.diag(cov_coef)**-0.5
    np.fill_diagonal(cor,1)
    return cor

class Server:
    def __init__(self,variables,confounders):
        self.variables = variables
        self.confounders = confounders
        self.client_names = []
        self.n_samples_per_cli = []
        self.global_genes = []
        
        self.XtX_glob = None
        self.Xty_glob = None
        self.beta = None
        self.cov_coef = None
        self.stdev_unscaled = None
        self.var = None
        self.sigma = None
        self.df_residual = None
        self.df_total = None
        self.Amean = None
        self.mean_logC = None
        self.lowess_curve = None
        self.results = None
        self.table = None
        
        self.mean_logC_w = None
        self.sigma_w = None
        self.lowess_curve_w = None
        
        # attributes for fed median
        self.lb = None
        self.ub = None
        self.approx_median = None # approximate median found in binary search
        self.tol = 0.1 # binary search of approx_medina stops when ub-lb<tol
        self.is_converged = None # whether approx_median is found for each gene
        self.le_gt = None # dataframe with numbers of values lesser or equal and greater than approx_median. "le":<= , "gt":>
        self.precise_median = None
        
    def join_client(self,cohort_name, client_genes,client_n_samples,client_conditions):
        '''Collects names of genes and conditions, and the numebr of samples from client'''
        if cohort_name in self.client_names:
            print("Choose client name other than",self.client_names, file = sys.stderr)
            print("Failed to join client",cohort_name, file = sys.stderr)
            return False
        for col in self.variables+self.confounders:
            if not col in client_conditions:
                print(col, "column is missing in the design matrix of",cohort_name, file = sys.stderr)
                print("Failed to join client",cohort_name, file = sys.stderr)
                return False
        
        if len(self.client_names) == 0:
            self.global_genes = sorted(client_genes)
        else:
            self.global_genes = [x for x in self.global_genes if x in client_genes]
        self.n_samples_per_cli.append(client_n_samples)
        
        self.client_names.append(cohort_name)
        print("Server: joined client  %s"%cohort_name)
        
        return True
    
    def filter_by_median(self, shuffled_count_matrices, threshold=1, func=np.median):
        '''Accepts a list of matrices with shuffled expressions and computes medians.'''
        medians = pd.concat(shuffled_count_matrices,axis=1).apply(lambda x :func(x),axis=1)
        keep_genes = list(medians[medians>threshold].index.values) # send to clients

        print("Genes with median counts < %s will be dropped:"%threshold,medians.shape[0] - len(keep_genes), file= sys.stderr) # with  median  CPM > 1 
        return keep_genes
    
    def compute_F(self,f):
        F = np.exp(np.mean(np.log(f)))
        return F
    
    ###### fedLmFit #####
    def compute_beta_and_beta_stdev(self,XtX_list, XtY_list):
        '''Calcualtes global beta and variance of beta'''
    
        k = len(self.variables+self.confounders)
        n = len(self.global_genes)
        self.XtX_glob = np.zeros((n,k,k))
        self.XtY_glob = np.zeros((n,k))
        self.stdev_unscaled = np.zeros((n,k))

        
        for i in range(0,len(self.client_names)):
            self.XtX_glob += XtX_list[i]
            self.XtY_glob += XtY_list[i]
        self.beta = np.zeros((n,k))
        self.rank = np.ones(n)*k
    
        for i in range(0,n):
            invXtX  = linalg.inv(self.XtX_glob[i,:,:])
            self.beta[i,:] = invXtX @ self.XtY_glob[i,:]
            self.stdev_unscaled[i,:] = np.sqrt(np.diag(invXtX )) #standart err for b coefficients 
            
    
    def aggregate_SSE_and_cov_coef(self,SSE_list,cov_coef_list, weighted=False):
        
        M= sum(self.n_samples_per_cli) # total number of samples
        n = len(self.global_genes)
        k = len(self.confounders+self.variables)
        self.SSE = np.zeros(n)
        self.cov_coef = np.zeros((k,k))
        
        for c in range(0,len(self.client_names)):
            self.cov_coef += cov_coef_list[c]
            for i in range(0,n):
                self.SSE[i] += SSE_list[c][i]
            
        self.cov_coef = linalg.inv(self.cov_coef )        
        # estimated residual variance
        self.var = self.SSE/(M-k)
        # estimated residual standard deviations
        if weighted:
            self.sigma_w =  np.sqrt(self.var) 
        else:
            self.sigma =  np.sqrt(self.var)
        # degrees of freedom
        self.df_residual = np.ones(n)*(M-k)

    ###### LOWESS curve fit ##########
    def compute_mean_logC(self,logC_conversion_term, weighted=False):
        M = np.sum(self.n_samples_per_cli)
        logC_conversion_term = logC_conversion_term/M -6*np.log2(10)
        self.Amean = self.Amean/M
        if weighted:
            self.mean_logC_w = self.Amean+logC_conversion_term
        else:
            self.mean_logC = self.Amean+logC_conversion_term
        
    
    def fit_LOWESS_curve(self,weighted=False):
        if weighted:
            mean_logC = self.mean_logC_w
            sigma  = self.sigma_w
        else:
            mean_logC = self.mean_logC
            sigma = self.sigma
        delta = (max(mean_logC) - min(mean_logC ))*0.01
        print("server: delta for LOWESS fit is set to:",delta)
        lowess = sm.nonparametric.lowess
        lowess_curve = lowess(sigma**0.5, mean_logC, frac=0.5, delta = delta, return_sorted=True,is_sorted=False)
        if weighted:
            self.lowess_curve_w = lowess_curve
        else:
            self.lowess_curve = lowess_curve

    ### apply contrasts
    
    def make_contrasts(self, contrasts=[]):
        '''Creates contrast matrix given deisgn matrix and pairs or columns to compare.\n
        For example:\n
        contrasts = [([A],[B]),([A,B],[C,D])] defines two contrasts:\n
        A-B and (A and B) - (C and D).'''
        df = {}
        conditions = self.variables + self.confounders
        for contr in contrasts:
            group1 , group2 = contr
            for name in group1+group2:
                if not name in conditions:
                    print(name, "not found in the design matrix.",file=sys.stderr)
                    exit(1)
            contr_name = "".join(map(str,group1))+"_vs_"+"".join(map(str,group2))
            c=pd.Series(data=np.zeros(len(conditions)),index=conditions)
            c[group1] = 1
            c[group2] = -1
            df[contr_name] = c
        return (pd.DataFrame.from_dict(df))  

    
    def fit_contasts(self,contrast_matrix):
        ncoef = self.cov_coef.shape[1]
        #	Correlation matrix of estimable coefficients
        #	Test whether design was orthogonal
        if not np.any(self.cov_coef):
            print("no coef correlation matrix found in fit - assuming orthogonal",file=sys.stderr)
            cormatrix = np.identity(ncoef)
            orthog = True
        else:
            cormatrix = cov2cor(self.cov_coef)
            if cormatrix.shape[0]*cormatrix.shape[1] < 2: 
                orthog = True
            else:
                if np.sum(np.abs(np.tril(cormatrix,k=-1))) < 1e-12:
                    orthog = True
                else:
                    orthog = False
        #print("is design orthogonal:",orthog)
        #	Replace NA coefficients with large (but finite) standard deviations
        #	to allow zero contrast entries to clobber NA coefficients.
        if np.any(np.isnan(self.beta)):
            print("Replace NA coefficients with large (but finite) standard deviations",file=sys.stderr)
            np.nan_to_num(self.beta,nan=0)
            np.nan_to_num(self.stdev_unscaled, nan=1e30)

        self.beta = self.beta.dot(contrast_matrix)
        # New covariance coefficiets matrix
        self.cov_coef = contrast_matrix.T.dot(self.cov_coef).dot(contrast_matrix)

        if orthog:
            self.stdev_unscaled = np.sqrt((self.stdev_unscaled**2).dot(contrast_matrix**2))
        else:
            n_genes = self.beta.shape[0]
            U = np.ones((n_genes, contrast_matrix.shape[1])) # genes x contrasts
            o = np.ones(ncoef)
            R = np.linalg.cholesky(cormatrix).T
            for i in range(0,n_genes):
                RUC = R @ (self.stdev_unscaled[i,] * contrast_matrix.T).T
                U[i,] = np.sqrt(o @ RUC**2)
            self.stdev_unscaled = U
            
    #### e-Bayes ############
    
    def trigamma(self,x):
        return polygamma(1,x)

    def psigamma(self,x,deriv=2):
        return polygamma(deriv,x)

    def trigammaInverse(self,x):

        if not hasattr(x, '__iter__'):
            x_ = np.array([x])
        for i in range(0,x_.shape[0]):
            if np.isnan(x_[i]):
                x_[i]= np.NaN
            elif x>1e7:
                x_[i] = 1./np.sqrt(x[i])
            elif x< 1e-6:
                x_[i] = 1./x[i]
        # Newton's method
        y = 0.5+1.0/x_
        for i in range(0,50):
            tri = self.trigamma(y)
            dif = tri*(1.0-tri/x_)/self.psigamma(y,deriv=2)
            y = y+dif
            if(np.max(-dif/y) < 1e-8): # tolerance
                return y

        print("Warning: Iteration limit exceeded",file=sys.stderr)
        return y

    def fitFDist(self,x, df1, covariate=False):
        '''Given x (sigma^2) and df1 (df_residual), fits x ~ scale * F(df1,df2) and returns estimated df2 and scale (s0^2)'''
        if covariate:
            # TBD
            print("Set covariate=False.", file=std.err)
            return
        # Avoid zero variances
        x = [max(x,0) for x in x]
        m = np.median(x)
        if(m==0):
            print("Warning: More than half of residual variances are exactly zero: eBayes unreliable", file=std.err)
            m = 1
        else:
            if 0 in x: 
                print("Warning: Zero sample variances detected, have been offset (+1e-5) away from zero", file=std.err)

        x = [max(x,1e-5 * m) for x in x] 
        z = np.log(x)
        e = z-digamma(df1*1.0/2)+np.log(df1*1.0/2)
        emean = np.nanmean(e)
        evar = np.nansum((e-emean)**2)/(len(x)-1)

        # Estimate scale and df2
        evar = evar - np.nanmean(self.trigamma(df1*1.0/2))
        
        if evar > 0:
            df2 = 2*self.trigammaInverse(evar)
            s20 = np.exp(emean+digamma(df2*1.0/2)-np.log(df2*1.0/2))
        else:
            df2 = np.Inf
            s20 = np.exp(emean)

        return s20,df2

    def posterior_var(self, var_prior=np.ndarray([]), df_prior=np.ndarray([])):
        var = self.var
        df=self.df_residual
        ndxs = np.argwhere(np.isfinite(var)).reshape(-1)
        # if not infinit vars
        if len(ndxs)==len(var): #np.isinf(df_prior).any():
            return (df*var + df_prior*var_prior) / (df+df_prior) # var_post  
        #For infinite df.prior, set var_post = var_prior
        var_post = np.repeat(var_prior,len(var))
        for ndx in ndxs:
            var_post[ndx] = (df[ndx]*var[ndx] + df_prior*var_prior)/(df[ndx]+df_prior)
        return var_post

    def squeezeVar(self, covariate=False, robust=False, winsor_tail_p=(0.05,0.1)):
        '''Estimates df and var priors and computes posterior variances.'''
        if robust:
            # TBD fitFDistRobustly()
            print("Set robust=False.",file=sys.stderr)
            return
        else:
            var_prior, df_prior = self.fitFDist(self.var, self.df_residual, covariate=covariate)

        if np.isnan(df_prior):
            print ("Error: Could not estimate prior df.",file=sys.stderr)
            return

        var_post = self.posterior_var(var_prior=var_prior, df_prior=df_prior)
        self.results = {"df_prior":df_prior,"var_prior":var_prior,"var_post":var_post}

    def moderatedT(self,covariate=False,robust=False, winsor_tail_p=(0.05,0.1)):
        #var,df_residual,coefficients,stdev_unscaled,
        self.squeezeVar(covariate=covariate, robust=robust, winsor_tail_p=winsor_tail_p)
        
        self.results["s2_prior"] = self.results["var_prior"]
        self.results["s2_post"] = self.results["var_post"]
        del self.results["var_prior"]
        del self.results["var_post"]
        self.results["t"] = self.beta / self.stdev_unscaled
        self.results["t"] = self.results["t"].T / np.sqrt( self.results["s2_post"])
        self.df_total = self.df_residual + self.results["df_prior"]
        df_pooled = sum(self.df_residual)
        self.df_total = np.minimum(self.df_total,df_pooled) # component-wise min

        self.results["p_value"] = 2*t.cdf(-np.abs(self.results["t"]),df=self.df_total)
        self.results["p_value"] = self.results["p_value"].T
        self.results["t"] = self.results["t"].T
        return self.results

    def tmixture_matrix(self,var_prior_lim=False,proportion=0.01):
        tstat = self.results["t"]
        stdev_unscaled = self.stdev_unscaled
        df_total = self.df_total
        ncoef = self.results["t"].shape[1]
        v0 = np.zeros(ncoef)
        for j in range(0,ncoef):
            v0[j] = self.tmixture_vector(tstat[:,j],stdev_unscaled[:,j],df_total,proportion,var_prior_lim)
        return v0

    def tmixture_vector(self,tstat,stdev_unscaled,df,proportion,var_prior_lim):
        ngenes = len(tstat)

        #Remove missing values
        notnan_ndx = np.where(~np.isnan(tstat))[0]
        if len(notnan_ndx) < ngenes:
            tstat = tstat[notnan_ndx]
            stdev_unscaled = stdev_unscaled[notnan_ndx]
            df = df[notnan_ndx]

        # ntarget t-statistics will be used for estimation

        ntarget = int(np.ceil(proportion/2*ngenes))
        if ntarget < 1: #
            return 

        # If ntarget is v small, ensure p at least matches selected proportion
        # This ensures ptarget < 1
        p = np.maximum(ntarget*1.0/ngenes,proportion)

        #Method requires that df be equal
        tstat = abs(tstat)
        MaxDF = np.max(df)
        i = np.where( df < MaxDF)[0]
        if len(i)>0:
            TailP = t.logcdf(tstat[i],df=df[i]) # PT: CDF of t-distribution: pt(tstat[i],df=df[i],lower.tail=FALSE,log.p=TRUE)
            # QT - qunatile funciton - returns a threshold value x 
            # below which random draws from the given CDF would fall p percent of the time. [wiki]
            tstat[i] = t.ppf(np.exp(TailP),df=MaxDF) # QT: qt(TailP,df=MaxDF,lower.tail=FALSE,log.p=TRUE)
            df[i] = MaxDF

        #Select top statistics
        order = tstat.argsort()[::-1][:ntarget] # TBD: ensure the order is decreasing
        tstat = tstat[order]
        v1 =  stdev_unscaled[order]**2


        #Compare to order statistics
        rank = np.array(range(1,ntarget+1))
        p0 =  2*t.sf(tstat,df=MaxDF) # PT
        ptarget = ((rank-0.5)/ngenes - (1.0-p)*p0) / p
        v0 = np.zeros(ntarget)
        pos = np.where(ptarget > p0)[0]
        if len(pos)>0:
            qtarget = -t.ppf(ptarget[pos]/2,df=MaxDF) #qt(ptarget[pos]/2,df=MaxDF,lower.tail=FALSE)
            #print(qtarget[:5])
            v0[pos] = v1[pos]*((tstat[pos]/qtarget)**2-1)

        if var_prior_lim[0] and var_prior_lim[1]:
            v0 = np.minimum(np.maximum(v0,var_prior_lim[0]),var_prior_lim[1])

        return np.mean(v0)
    
    
    def Bstat(self,stdev_coef_lim = np.array([0.1,4]),proportion = 0.01):

        var_prior_lim  = stdev_coef_lim**2/np.median(self.results["s2_prior"])
        #print("Limits for var.prior:",var_prior_lim)

        self.results["var_prior"] = self.tmixture_matrix(proportion=0.01,var_prior_lim=var_prior_lim)

        nan_ndx = np.argwhere(np.isnan(self.results["var_prior"]))
        if len(nan_ndx)>0:
            self.results["var.prior"][ nan_ndx] <- 1.0/self.results["s2_prior"]
            print("Warning: Estimation of var.prior failed - set to default value",file = sys.stderr)
        r = np.outer(np.ones(self.results["t"].shape[0]),self.results["var_prior"])
        r = (self.stdev_unscaled**2+r) / self.stdev_unscaled**2
        t2 = self.results["t"]**2

        valid_df_ndx = np.where(self.results["df_prior"] <= 1e6)[0]
        if len(valid_df_ndx)<len(self.results["df_prior"]):
            print("Large (>1e6) priors for DF:", len(valid_df_ndx))
            kernel = t2*(1-1.0/r)/2
            for i in valid_df_ndx:
                kernel[i] = (1+self.df_total[i])/2*np.log((t2[i,:].T+self.df_total[i]) / ((t2[i,:]/r[i,:]).T+self.df_total[i]))
        else:
            kernel = (1+self.df_total)/2*np.log((t2.T+self.df_total)/((t2/r).T+self.df_total))

        self.results["lods"] = np.log(proportion/(1-proportion))-np.log(r)/2+kernel.T
    
    def topTableT(self,adjust="fdr_bh", p_value=1.0,lfc=0,confint=0.95):
        feature_names = self.global_genes
        self.results["logFC"] = pd.Series(self.beta[:,0],index=feature_names)
        
        # confidence intervals for LogFC
        if confint:
            alpha = (1.0+confint)/2
            margin_error = np.sqrt(self.results["s2_post"]) *self.stdev_unscaled[:,0] * t.ppf(alpha, df=self.df_total)
            self.results["CI.L"] = self.results["logFC"]-margin_error
            self.results["CI.R"] = self.results["logFC"] +margin_error
        # adjusting p-value for multiple testing
        if_passed, adj_pval,alphacSidak,alphacBonf = multipletests(self.results["p_value"][:,0], alpha=p_value, method=adjust,
                                           is_sorted=False, returnsorted=False)
        self.results["adj.P.Val"] = pd.Series(adj_pval,index=feature_names)
        self.results["P.Value"] = pd.Series(self.results["p_value"][:,0],index=feature_names)
        # make table 
        self.table = copy(self.results)
        # remove 'df_prior', 's2_prior', 's2_post', 'df_total','var_prior'
        for key in ['df_prior', 's2_prior', 's2_post', 'var_prior',"p_value"]:        
            del self.table[key]
        self.table["t"] = pd.Series(self.table["t"][:,0],index=feature_names)
        self.table["lods"] = pd.Series(self.table["lods"][:,0],index=feature_names)
        self.table = pd.DataFrame.from_dict(self.table)
    
    def eBayes(self):
        covariate = False # Amean for limma-trend
        robust = False # 
        winsor_tail_p = (0.05,0.1) # needed for fitFDistRobustly()

        var = self.sigma**2 
        self.results = self.moderatedT(covariate=covariate,robust=robust, winsor_tail_p=winsor_tail_p)
        #self.results = moderatedT(self.var,self.df_residual,self.beta,self.stdev_unscaled,
        #                         covariate=covariate,robust=robust, winsor_tail_p=winsor_tail_p)
        self.results["AveExpr"] = self.Amean
        
        self.Bstat(stdev_coef_lim = np.array([0.1,4]),proportion = 0.01)
        
        self.topTableT(adjust="fdr_bh", p_value=1.0,lfc=0,confint=0.95)
        self.table = self.table.sort_values(by="P.Value")
        
        
    def plot_voom_results(self,svg_file=""):
        '''Plots relationships between average log-count and square root of estimated std.dev for each gene.'''
        fig, axes = plt.subplots(1, 2, figsize=(15,5), sharey=True)
        axes[0].set_title("Before Voom")
        axes[0].scatter(x=self.mean_logC, y=self.sigma**0.5,s=0.5,color="black")
        axes[0].set_xlabel(r'average($log_2$(counts + 0.5)) per gene')
        axes[0].set_ylabel(r'$\sqrt{s_g}$ - root of estimated std.dev for gene g')
        axes[0].scatter(self.lowess_curve[:,0],self.lowess_curve[:,1],s=0.5,color="red")

        axes[1].set_title("After Voom")
        axes[1].scatter(x=self.mean_logC_w, y=self.sigma_w**0.5,s=0.5,color="black")
        axes[1].set_xlabel(r'average($log_2$(counts + 0.5)) per gene')
        #axes[1].set_ylabel(r'$\sqrt{s_g}$ - root of estimated std.dev for gene g')
        axes[1].scatter(self.lowess_curve_w[:,0],self.lowess_curve_w[:,1],s=0.5,color="red")
        if svg_file:
            plt.savefig(svg_file)