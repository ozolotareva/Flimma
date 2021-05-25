# usage: Rscript run_MA.R working_dir/ cli1 cli2 ...
library("MetaVolcanoR")
#https://bioconductor.org/packages/release/bioc/vignettes/MetaVolcanoR/inst/doc/MetaVolcano.html#random-effect-model-metavolcano

args <- commandArgs(trailingOnly = TRUE)
w_dir <- args[1] #
cohorts <- args[2:length(args)] # all but 1st arguments are cohort names

diffexplist <- list()

for (cohort in cohorts){ #,"Other")){
    fname <- paste0(w_dir,cohort,".Rlimma_table.tsv")
    res <- read.table(fname, row.names = 1,sep="\t",dec=",")
    res["Symbol"] <- rownames(res)
    diffexplist[[cohort]] <- res
}

suppressWarnings(meta_degs_comb <- combining_mv(diffexp=diffexplist,
                   pcriteria="P.Value",
                   foldchangecol='logFC', 
                   genenamecol='Symbol',
                   geneidcol=NULL,
                   metafc='Mean',
                   metathr=0.01, 
                   collaps=TRUE,
                   jobname="MetaVolcano",
                   outputfolder=".",
                   draw='HTML'))
result <- meta_degs_comb@metaresult
result <- result[order(result["metap"]),]
write.table(result,paste0(w_dir,"/MA_CM.tsv"),row.names=TRUE,sep="\t", quote = FALSE,dec = ".")
#head(result,3)

suppressWarnings(meta_degs_rem <- rem_mv(diffexp=diffexplist,
            pcriteria="P.Value",
            foldchangecol='logFC', 
            genenamecol='Symbol',
            geneidcol=NULL,
            collaps=FALSE,
            llcol='CI.L',
            rlcol='CI.R',
            vcol=NULL, 
            cvar=TRUE,
            metathr=0.01,
            jobname="MetaVolcano",
            outputfolder=".", 
            draw='HTML',
            ncores=1))

result <- meta_degs_rem@metaresult
result  <- result[order(result["randomP"]),]
write.table(result,paste0(w_dir,"/MA_REM.tsv"),row.names=TRUE,sep="\t", quote = FALSE,dec = ".")
#head(result,3)

