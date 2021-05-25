# conda activate r_microarrays
# /home/olya/anaconda3/envs/r_microarrays/bin/Rscript run_MetaDE_and_RankProd.R working_dir/ cli1 cli2 ...

args <- commandArgs(trailingOnly = TRUE)
w_dir <- args[1] #
cohorts <- args[2:length(args)] # all but 1st arguments are cohort names

suppressPackageStartupMessages(library("MetaDE"))

log2FC <- data.frame()
pvals <- data.frame()

for (cohort in cohorts){ #,"Other")){
    fname <- paste0(w_dir,cohort,".Rlimma_table.tsv")
    res <- read.table(fname, row.names = 1,sep="\t",dec=",")
    lfc <- res["logFC"]
    pv <- res["P.Value"]
    #res$rank <- c(1:dim(res)[[1]])
    #rank <- res["rank"]
    if (dim(log2FC)[[2]] == 0) {
        log2FC <- lfc
        pvals <- pv
    }
    else{
     lfc <- lfc[rownames(log2FC),]       
     log2FC <-  cbind(log2FC, lfc)
     pv <- pv[rownames(pvals),] 
     pvals <-  cbind(pvals, pv)
    }
}

colnames(log2FC) <- cohorts
colnames(pvals) <- cohorts
log2FC <- as.matrix(log2FC)
pvals <- as.matrix(pvals)

ind_res <- list("log2FC" = log2FC,"p"=pvals)


methods <- c("Stouffer","Fisher","roP")
for(i in 1:3){ # Stouffer,roP(rth=2), Fisher
    meta.method <- methods[i]
    print(meta.method)
    rth <- NULL
    if(meta.method == "roP"){rth<-max(2,length(cohorts) %/% 2)}
    meta.res <- MetaDE.pvalue(ind_res,meta.method,rth=rth,parametric=T)
    result <- data.frame(ind.p = meta.res$ind.p,
                          stat = meta.res$meta.analysis$stat,
                          pval = meta.res$meta.analysis$pval,
                          FDR = meta.res$meta.analysis$FDR,
                          weight = meta.res$meta.analysis$AW.weight
                         )
    colnames(result)[seq(length(cohorts)+1,length(cohorts)+3)] <- c("stat","pval","FDR")
    write.table(result,paste0(w_dir,"/MA_",meta.method,".tsv"),row.names=TRUE,sep="\t", quote = FALSE,dec = ".")
}

