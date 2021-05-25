import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys,os
from statsmodels.stats.multitest import multipletests

from time import time

from matplotlib.colors import ListedColormap as lcmap

cmap = lcmap(["blue","skyblue","red"], name='from_list', N=None)

def read_results(workdir):
    df = {}
    offset = np.finfo(np.float).tiny # replace zero pvals with this value
    
    rlimma = pd.read_csv(workdir+"/All.Rlimma_table.tsv",sep="\t")
    rlimma = rlimma.applymap(lambda x: float(x.replace(",",".")))
    rlimma.loc[rlimma["adj.P.Val"]==0,"adj.P.Val"] = offset
    df["pv_Rlimma"] = -np.log10(rlimma["adj.P.Val"])
    df["lfc_Rlimma"] = rlimma["logFC"]

    ma_cm = pd.read_csv(workdir+"/MA_CM.tsv",sep="\t")
    ma_cm.index = ma_cm["Symbol"].values
    df["lfc_Fisher"] = ma_cm["metafc"]
    if_passed, adj_pval,alphacSidak,alphacBonf = multipletests(ma_cm["metap"].values, alpha=1.0, method='fdr_bh',
                                           is_sorted=False, returnsorted=False)
    adj_pval[adj_pval==0] = offset
    df["pv_Fisher"] = -np.log10(pd.Series(adj_pval,index=ma_cm["metap"].index))


    ma_rem = pd.read_csv(workdir+"/MA_REM.tsv",sep="\t")
    ma_rem.index = ma_rem["Symbol"].values

    df["lfc_REM"] = ma_rem["randomSummary"]
    if_passed, adj_pval,alphacSidak,alphacBonf = multipletests(ma_rem["randomP"].values, alpha=1.0, method='fdr_bh',
                                           is_sorted=False, returnsorted=False)
    adj_pval[adj_pval==0] = offset
    df["pv_REM"] = -np.log10(pd.Series(adj_pval,index=ma_rem["randomP"].index))

    flimma = pd.read_csv(workdir+"/All.flimma_results.tsv",sep="\t",index_col=0)
    flimma.loc[flimma["adj.P.Val"]==0,"adj.P.Val"] = offset
    df["pv_Flimma"] = -np.log10(flimma["adj.P.Val"])
    df["lfc_Flimma"] = flimma["logFC"]

    ### Stoufer 
    stoufer  = pd.read_csv(workdir+"/MA_Stouffer.tsv",sep="\t",index_col=0)
    stoufer.loc[stoufer["FDR"]==0,"FDR"] = offset
    df["pv_Stouffer"] = -np.log10(stoufer["FDR"])
    df["lfc_Stouffer"] = df["lfc_Fisher"]  # take logFC from MetaVolcanoR
    ### RankProd
    rankprod  = pd.read_csv(workdir+"/MA_RankProd.tsv",sep="\t",index_col=0)
    rankprod["FDR"] = rankprod.loc[:,["down_reg.FDR","up_reg.FDR"]].min(axis=1)
    rankprod.loc[rankprod["FDR"]==0,"FDR"] = offset
    df["pv_RankProd"] = -np.log10(rankprod["FDR"])
    df["lfc_RankProd"] = df["lfc_Fisher"]  # take logFC from MetaVolcanoR
    
    df = pd.DataFrame.from_dict(df)
    df = df.dropna(axis=0)
    return df

def make_confusion_matrix(df,lfc_thr=2,adj_pval_thr = 0.05,
                          methods=["Flimma","Fisher","Stouffer","REM","RankProd"],top_genes=0):
    confusions={}
    all_genes = set(df.index.values)
    DE = df.loc[df["pv_Rlimma"]>-np.log10(adj_pval_thr),:]
    DE = set(list(DE.loc[DE["lfc_Rlimma"]>lfc_thr,:].index.values)+
             list(DE.loc[DE["lfc_Rlimma"]<-lfc_thr,:].index.values))
    not_DE = all_genes.difference(DE)
    #prnt("T:",len(T), "F:",len(F))
    for m in methods:
        P = df.loc[df["pv_"+m]>-np.log10(adj_pval_thr),:]
        P = set(list(P.loc[P["lfc_"+m]>lfc_thr,:].index.values)+list(P.loc[P["lfc_"+m]<-lfc_thr,:].index.values))
        N = all_genes.difference(P)
        TP=len(DE.intersection(P))
        FP = len(not_DE.intersection(P))
        TN = len(not_DE.intersection(N))
        FN = len(DE.intersection(N))
        if (TP+FP)>0:
            Prec = TP*1.0/(TP+FP)
        else:
            Prec =0
        if (TP+FN) >0:
            Rec = TP*1.0/(TP+FN)
        else:
            Rec = 0
        if Prec and Rec:
            F1 = 2* (Prec*Rec)/(Prec+Rec)
        else:
            F1=0
            
        confusions[m] = {"TP":TP,"FP":FP,
                        "TN":len(not_DE.intersection(N)),"FN":len(DE.intersection(N)),
                         "Precision":Prec,"Recall":Rec, "F1":F1}
        # RMSE for -log10 p-values
        if top_genes>0:
            d = df.sort_values(by="pv_Rlimma",ascending = False).head(top_genes)
        else:
            d = df
        x = d["pv_Rlimma"].values
        y = d["pv_"+m].values
        rmse = np.sqrt(np.sum((x-y)**2)/len(x))
        confusions[m]["RMSE"] = rmse
        
    confusions = pd.DataFrame.from_dict(confusions).T   
    # correlation of all p-values
    corrs = d[["pv_"+"Rlimma"]+["pv_"+m for m in methods]].corr().loc[["pv_"+"Rlimma"],]
    corrs.rename(lambda x: x.replace("pv_",""), axis="columns",inplace = True)
    corrs = corrs.T['pv_Rlimma']
    rank_corrs = d[["pv_"+"Rlimma"]+["pv_"+m for m in methods]].corr(method="spearman").loc[["pv_"+"Rlimma"],]
    rank_corrs.rename(lambda x: x.replace("pv_",""), axis="columns",inplace = True)
    rank_corrs = rank_corrs.T['pv_Rlimma']
    confusions["r"] = corrs
    confusions["ρ"] = rank_corrs
    confusions= confusions.T
    return confusions.loc[["TP","TN","FP","FN","Precision","Recall","F1","r","ρ","RMSE"],:]

def plt_results(dfs, methods = ["Flimma","Fisher","Stouffer","REM","RankProd"],
                colors = ["red","blue","cyan","lightgreen","grey"], 
                what="pv_", suptitle="$-log_{10}(adj.p-values)$", text="",dotsize=1):
    fig, axes = plt.subplots(1, 3, figsize=(17,5), sharey=False)
    #fig.suptitle(suptitle,fontsize=16)
    i=0
    se = 0
    results = {}
    for k in ["Balance","Mild Imbalance","Strong Imbalance"]:
        df = dfs[k]
        axes[i].set_title(k,fontsize=16)
        rmse = {}
        
        for j in range(len(methods)):
            method = methods[j]
            col = colors[j]
            x = df[what+"Rlimma"].values
            y = df[what+method].values
            rmse[method] = np.sqrt(np.sum((x-y)**2)/len(x))
            axes[i].scatter(x = x,
                            y= y,s=dotsize, color=col,alpha=0.5)
            
        axes[i].set_xlabel('limma voom on complete dataset',fontsize=14)
        axes[i].set_ylabel('other methods',fontsize=14)
        axes[i].plot([np.min(df.values),np.max(df.values)+5],[np.min(df.values),np.max(df.values)+5],
                     color = "red",ls="--",lw=0.1)
        
        corrs = df[[what+"Rlimma"]+[what+m for m in methods]].corr().loc[[what+"Rlimma"],]
        corrs.rename(lambda x: x.replace(what,""), axis="columns",inplace = True)
        corrs = corrs.T.to_dict()[what+'Rlimma']
        rank_corrs = df[[what+"Rlimma"]+[what+m for m in methods]].corr(method="spearman").loc[[what+"Rlimma"],]
        rank_corrs.rename(lambda x: x.replace(what,""), axis="columns",inplace = True)
        rank_corrs = rank_corrs.T.to_dict()[what+'Rlimma']
        patches = {}
        for j in range(len(methods)):
            method = methods[j]
            col = colors[j]
            r = corrs[method]
            rho = rank_corrs[method]
            err = rmse[method]
            if err <0.01 or err>100:
                err = "{:.2e}".format(err)
            else:
                err = round(err,2)
            patch = mpatches.Patch(color=col, label='%s: r=%s; ρ=%s; RMSE=%s'%(method, round(r,2),round(rho,2),err))
            patches[method] = patch 
        axes[i].legend(handles=[patches[m] for m in methods],loc='upper left',)
        i+=1
        results[(k,"r")] = corrs
        results[(k,"ρ")] = rank_corrs
        results[(k,"RMSE")] = pd.Series(rmse)
    results = pd.DataFrame.from_dict(results)
    if text:
        tmp = axes[0].text(-0.2*np.max(df.values), np.max(df.values), text, fontsize=24)
    return results.loc[methods,]
    

def calc_stats(df,lfc_thr=1,adj_pval_thr = -np.log10(0.05),
               stats=["TP","TN","FP","FN","Precision","Recall","F1","r","ρ","RMSE"],
               methods=["Flimma","Fisher","Stouffer","REM","RankProd"],top_genes=-1):
    results={}
    all_genes = set(df.index.values)

    if top_genes<=0:
        top_genes = df.shape[0]
    #de = df.sort_values(by="pv_Rlimma",ascending = False)
    de = df.loc[df["pv_Rlimma"]>adj_pval_thr,:]
    de = de.loc[np.abs(de["lfc_Rlimma"])>=lfc_thr,:].head(top_genes)
    
    # truth: DE and not DE genes predicted by limma
    T = set(de.index.values)
    F = all_genes.difference(T)
    #prnt("T:",len(T), "F:",len(F))
    if len(set(stats).intersection(set(["TP","TN","FP","FN","Precision","Recall","F1"])))>0:
        for m in methods:
            # prediction
            de2 = df.loc[:,["pv_"+m,"lfc_"+m]].sort_values(by="pv_"+m,ascending = False)
            de2 = de2.loc[de2["pv_"+m]>adj_pval_thr,:]
            de2 = de2.loc[np.abs(de2["lfc_"+m])>=lfc_thr,:].head(top_genes)
            P = set(de2.index.values)
            N = all_genes.difference(P)
            
            TP=len(T.intersection(P))
            FP = len(F.intersection(P))
            TN = len(F.intersection(N))
            FN = len(T.intersection(N))
            if (TP+FP)>0:
                Prec = TP*1.0/(TP+FP)
            else:
                Prec =0
            if (TP+FN) >0:
                Rec = TP*1.0/(TP+FN)
            else:
                Rec = 0
            if Prec and Rec:
                F1 = 2* (Prec*Rec)/(Prec+Rec)
            else:
                F1=0

            results[m] = {"TP":TP,"FP":FP,
                            "TN":TN,"FN":FN,
                             "Precision":Prec,"Recall":Rec, "F1":F1}

           
    # correlation of all p-values
    if "RMSE" in stats:
        for m in methods:
            # RMSE for -log10 p-values
            df = df.sort_values(by="pv_Rlimma",ascending = False).head(top_genes)
            x = df["pv_Rlimma"].values
            y = df["pv_"+m].values
            rmse = np.sqrt(np.sum((x-y)**2)/len(x))
            if m in results.keys():
                results[m]["RMSE"] = rmse
            else:
                results[m] = {"RMSE":rmse}
    # turn results to df if it is not empty
    if len(results.keys())>0:
        results = pd.DataFrame.from_dict(results).T
    if "r" in stats:
        df = df.sort_values(by="pv_Rlimma",ascending = False).head(top_genes)
        corrs = df[["pv_"+"Rlimma"]+["pv_"+m for m in methods]].corr().loc[["pv_"+"Rlimma"],]
        corrs.rename(lambda x: x.replace("pv_",""), axis="columns",inplace = True)
        corrs = corrs.loc[:,methods]
        corrs = corrs.T['pv_Rlimma']
        results["r"] = corrs
    if "ρ" in stats: 
        df = df.sort_values(by="pv_Rlimma",ascending = False).head(top_genes)
        rank_corrs = df[["pv_"+"Rlimma"]+["pv_"+m for m in methods]].corr(method="spearman").loc[["pv_"+"Rlimma"],]
        rank_corrs.rename(lambda x: x.replace("pv_",""), axis="columns",inplace = True)
        rank_corrs = rank_corrs.loc[:,methods]
        rank_corrs = rank_corrs.T['pv_Rlimma']
        results["ρ"] = rank_corrs
    
    # turn results to df if it is still a dict
    if type(results)==dict:
        results = pd.DataFrame.from_dict(results)
    return results.loc[:,stats]

def plot_stats_for_topN(dfs,datasets = ["Balance","Mild Imbalance","Strong Imbalance"],
                        metrics=["F1"],
                        methods = ["Flimma","Fisher","Stouffer","REM","RankProd"],
                  colors = ["red","blue","cyan","lightgreen","grey"],
                  min_n_genes=10,max_n_genes = 1000,step=10, text="",log=False,figfile= "",suptitle=""):
    
    """Calculated and plots statisctics for top N genes ordered by p-value. 
   Top genes are chosen based on a sliding threshold, starting from 'min_n_genes' and moving to 'max_n_genes' with 'step'."""
    cmap = lcmap(colors, name='from_list', N=None)
    fig, all_axes = plt.subplots(len(metrics), 3, figsize=(17,5*len(metrics)), sharey=False)
    all_stats ={}
    for k in range(len(metrics)):
        metric = metrics[k]
        all_stats[metric]={}
        if len(metrics)==1:
            axes = all_axes
        else:
            axes = all_axes[k]
        for i in range(len(datasets)):
            ds = datasets[i]
            df = dfs[ds]
            df = df.sort_values(by="pv_Rlimma",ascending = False)
            n_genes = df.shape[0]
            stats  = {}
            top_n_genes = np.arange(min_n_genes,max_n_genes,step)
            for j in range(len(top_n_genes)): #
                confusion_matrix = calc_stats(df,lfc_thr=1.0,adj_pval_thr = -np.log10(0.05),stats=[metric],
                                                         methods=methods,top_genes=top_n_genes[j])
                stats[top_n_genes[j]] = confusion_matrix[metric]
            stats = pd.DataFrame.from_dict(stats)
            tmp = stats.T.plot(ax =  axes[i],cmap = cmap)
            if log:
                axes[i].set_yscale('log')
            #print(stats)
            if i==1 and k==len(metrics)-1:
                tmp = axes[i].set_xlabel("number of top-ranked genes",fontsize=14)
            if i ==0:
                if log:
                    tmp = axes[i].set_ylabel("$log_{10}($"+metric+"$)$",fontsize=24)
                else:
                    tmp = axes[i].set_ylabel(metric,fontsize=18)
                if text:
                    tmp = axes[0].text(-0.15*max_n_genes, np.max(stats.values)*1.0, text, fontsize=24)
            if i >0 or k!=len(metrics)-1:
                axes[i].get_legend().remove()
            if k==0:
                tmp = axes[i].set_title(ds,fontsize=16)
            all_stats[metric][ds] = stats
    if suptitle:
        fig.suptitle(suptitle,fontsize=24)
    if figfile:
        fig.savefig(figfile)
    return all_stats


def find_diff(dfs):
    stats = {}
    for ds in dfs.keys():
        df = dfs[ds]
        diff = np.abs(df["pv_Rlimma"] - df["pv_Flimma"] )
        
        stats[ds] ={ ("-log10(p-value)","min") : np.min(diff),
                    ("-log10(p-value)","mean"):np.mean(diff),
                    ("-log10(p-value)","max"):np.max(diff)}
        #print(ds, np.mean(diff))
        diff = np.abs(df["lfc_Rlimma"] - df["lfc_Flimma"] )
        stats[ds].update({ ("log2(FC)","min") : np.min(diff),
                    ("log2(FC)","mean"):np.mean(diff),
                    ("log2(FC)","max"):np.max(diff)})
    return pd.DataFrame.from_dict(stats).T