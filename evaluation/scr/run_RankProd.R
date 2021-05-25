# Rscript run_MetaDE_and_RankProd.R working_dir/ cli1 cli2 ...
suppressPackageStartupMessages(library(RankProd))

args <- commandArgs(trailingOnly = TRUE)
w_dir <- args[1] #
cohorts <- args[2:length(args)] # all but 1st arguments are cohort names

lfcs <- data.frame()
for (cohort in cohorts){ 
    fname <- paste0(w_dir,cohort,".Rlimma_table.tsv")
    res <- read.table(fname, row.names = 1,sep="\t",dec=",")
    lfc <- res["logFC"]
    if (dim(lfcs)[[2]] == 0) {
        lfcs <- lfc
    }
    else{
     lfc <- lfc[rownames(lfcs),]       
     lfcs<-  cbind(lfcs, lfc)
    }
}
colnames(lfcs) <- cohorts
lfcs <- as.matrix(lfcs)

classes <- rep(1, length(cohorts))
origins <- rep(1, length(cohorts))#c(1:length(cohorts))

RP_obj_lfc <-  RP.advance(lfcs,classes,origins, logged=TRUE,gene.names=rownames(lfcs),rand=1)

pvals <- RP_obj_lfc$pval
colnames(pvals) <- c("down_reg.pval","up_reg.pval")
fdr <- RP_obj_lfc$pval
colnames(fdr) <- c("down_reg.FDR","up_reg.FDR")
avgL2FC <- RP_obj_lfc$AveFC
colnames(avgL2FC) <- c("avgL2FC")
result <- cbind(pvals,fdr,avgL2FC)
write.table(result,paste0(w_dir,"/MA_RankProd.tsv"),row.names=TRUE,sep="\t", quote = FALSE,dec = ".")