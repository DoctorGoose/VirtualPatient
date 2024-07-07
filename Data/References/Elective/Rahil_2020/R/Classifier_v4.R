##Classifier to diagnose viral shedding post-challenge
##The model was built using data from cytokine/chemokines, CBC, cell abundance, and cell activation and proliferation (median CD38 and Ki-67 expression)


##set your datapath to the folder which hosts the data
## set your output to the folder which will  host results
datapath='/Users/zainabrahil/Desktop/testresult/ElasticNet'
outpath='/Users/zainabrahil/Desktop/testresult'
datasets=c( 'Luminex_3'  , 'CBC_3', 'Cell Quantity Reordered_06MAY2020', 'Medians_06MAY2020-colremoved' )
datanames=c('Luminex', 'CBC', 'Cell Quantity', 'Medians')##Name these data cells
outcomes=read.csv(sprintf('%s/%s.csv', datapath, 'Outcomes binary_11JUN2018'), header=TRUE)
hai=outcomes[,10]

i=1
temp=read.csv(sprintf('%s/%s.csv', datapath, datasets[i]))

data=list()
for (i in seq(length(datasets))){
    data[[i]]=read.csv(sprintf('%s/%s.csv', datapath, datasets[i]), header=TRUE)
}

freqs=data[[1]]
patients=freqs[,1]
times=freqs[,2]
virus=freqs[,3]

temp=colnames(freqs)
for (i in seq(length(data)))
    data[[i]]=data[[i]][,-c(1:3)]

utimes=c(1:8, 29, 60)
upatients=unique(patients)



dataarr=list()
for (d in seq(length(data))){
    dataarr[[d]]=array(NA, c(length(upatients), length(utimes), ncol(data[[d]])))
    for (t in seq(utimes)){
        print(t)
        for (p in seq(length(upatients))){
            dataarr[[d]][p,t,]=unlist(data[[d]][which((patients==(upatients[p])) & (times==(utimes[t]))),])
            dimnames(dataarr[[d]])=list(upatients, utimes, colnames(data[[d]]))
        }
    }
    if(FALSE){
        temp=dataarr[[d]]
        nas=which(is.na(dataarr[[d]]), arr.ind=TRUE)
        if (nrow(nas)>0){
            for (i in seq(nrow(nas))){
                dataarr[[d]][nas[i,1], nas[i,2], nas[i,3]]=mean(temp[nas[i,1],,], na.rm=TRUE)
            }
        }
    }
}



datamat=dataarr
for (d in seq(length(data))){
    dim(datamat[[d]])=c(length(upatients)* length(utimes), ncol(data[[d]]))
    rownames(datamat[[d]])=sprintf('%s_%s', expand.grid(dimnames(dataarr[[d]])[[1]], dimnames(dataarr[[d]])[[2]])[,1], expand.grid(dimnames(dataarr[[d]])[[1]], dimnames(dataarr[[d]])[[2]])[,2])
    colnames(datamat[[d]])=colnames(data[[d]])
    if(TRUE){
        temp=datamat[[d]]
        nas=which(is.na(datamat[[d]]), arr.ind=TRUE)
        if (nrow(nas)>0){
            for (i in seq(nrow(nas))){
                datamat[[d]][nas[i,1], nas[i,2]]=mean(temp[,nas[i,2]], na.rm=TRUE)
            }
        }
    }
}


pvirus=(virus[match(upatients, patients)]==1)
phai=(hai[match(upatients, patients)]==1)
library(randomForest)
library(pROC)

rowMedians <- function(x){
    ans=vector()
    for (i in seq(nrow(x)))
        ans[i]=median(x[i,], na.rm=TRUE)
    return(ans)
}


modelpvals=matrix(NA, length(utimes), length(datamat))
preds=array(NA, c(length(utimes), length(datamat), length(upatients)))
importance=list()
pvalues=list()
ansmat_list=list()
for (z in seq(length(utimes))){
    importance[[z]]=list()
    pvalues[[z]]=list()
    print(z/length(utimes))
    index=vector()
    #iterations set to 100
    ansmat_list[[z]]=array(NA,c(100, length(upatients), length(datamat)))
    
    for (i in seq(length(upatients))){
        index[i]=which(rownames(datamat[[1]])==sprintf('%s_%d',upatients[i], utimes[z]))
    }

    pdatamat=list()
    for (i in seq(length(datamat)))
        pdatamat[[i]]=datamat[[i]][index,]

    for(d in seq(length(pdatamat))){        
        print(d)
        X=pdatamat[[d]]
        Y=factor(pvirus)
        index=which(!is.na(rowMeans(X)))
        X=X[index,]
        Y=Y[index]
        if(length(index)==0){
            modelpvals[z,d]=1
            preds[z,d,]=rep(NA, length(pvirus))
            next
        }
        
        if (min(table(Y))==0){
            modelpvals[z,d]=1
            preds[z,d,]=rep(NA, length(pvirus))
            next
        }        
        #iterations set to 100
        ansmat=matrix(NA, 100, length(Y))
        imps=rep(0, ncol(X))
        for (iter in seq(nrow(ansmat))){
            train=sample(seq(length(Y)), length(Y)*4/5)
            #train=sample(seq(length(Y)), length(Y)*2/3)
            if (min(table(Y[train]))==0)
                next
            test=seq(length(Y))[-train]
            mod=randomForest(X[train,], Y[train]) 
            test_=predict(mod, X[test,], type='prob')
            ansmat[iter, test]=predict(mod, X[test,], type='prob')[,2]
            imps=imps+importance(mod)
        }
        imps=imps/nrow(ansmat)

        pvs=vector()
        for (iter in seq(ncol(X))){
            pvs[iter]=wilcox.test(X[,iter] ~ Y)$p.value
        }

        importance[[z]][[d]]=imps
        pvalues[[z]][[d]]=pvs
        ans=rowMedians(t(ansmat))

        modelpvals[z,d]=wilcox.test(ans ~ Y)$p.value
        preds[z,d,index]=ans
        ansmat_list[[z]][,,d]=ansmat
    }
}


###initializing vector for combined pvalues
combinedpvals=vector()
for (z in seq(length(utimes))){
  ans=rowMedians(t(preds[z,,]))
  if(length(table(ans))==0){
    combinedpvals[z]=1
    next
  }
  combinedpvals[z]=wilcox.test(ans ~ pvirus)$p.value
}


##outputing daywise heatmap based on model p-values
png(sprintf('%s/heatmap.png', outpath), width=960)
library(gplots)
temp=matrix(p.adjust(modelpvals), nrow=nrow(modelpvals))
heatmap.2(t(-log10(temp)), Colv=NA, Rowv=NA, scale='none', trace='none', labRow=datanames, labCol=utimes, margins=c(5, 10), key.xlab='-log10(pvalue)')
dev.off()


pdf(sprintf('%s/heatmap.pdf', outpath))
library(gplots)
library(RColorBrewer)
cpvals=cbind(modelpvals, combinedpvals)
cdatanames=c(datanames, 'Combined')
temp=matrix(p.adjust(cpvals, method = 'fdr'), nrow=nrow(modelpvals))
Colors=brewer.pal(8,"Blues")
Colors=colorRampPalette(Colors)(120)
heatmap.2(t(-log10(temp[1:8,])), col=Colors, ,Colv=NA, Rowv=NA, scale='none', trace='none', labRow=cdatanames, labCol=utimes[1:8], margins=c(5, 10), key.xlab='-log10(pvalue)', density.info = 'none')
dev.off()
colnames(temp)=cdatanames
rownames(temp)=utimes
write.csv(temp, sprintf('%s/sameday_virus_test_train.csv', outpath))


ansmat_list_combine=array(NA,c(nrow(ansmat), length(upatients), length(utimes)))
for( z in seq(utimes)){
  for(r in seq(nrow(ansmat))){
    ansmat_list_combine[,,z][r,]= rowMedians(ansmat_list[[z]][r,,])
  }
}
##outputing daywise-ROC
out= ansmat_list_combine
library(matrixStats)
library(ROCR)
for( d in seq(length(utimes))){
  day=d
  out_1=out[,,d]
  pred_= colMedians(out_1, na.rm = TRUE)
  pred_=pred_
  act=pvirus
  ##save name
  save_name=paste0('%s/ROC_1-3split',utimes[day], '.pdf')
  pdf(sprintf(save_name, outpath))
  par(pty="s")
  plot(0:1, 0:1, type="n", xlab="False positive rate", ylab="True positive 
    rate") 
  abline(0, 1, col="red") 
  pred=prediction(pred_,act)
  perf <- performance(pred, "tpr", "fpr") 
  plot(perf, lwd=2 ,add=TRUE) 
  auc.perf = performance(pred, measure = "auc")
  auc=auc.perf@y.values[[1]]
  legend("bottomright", legend=c(paste0('AUC = ', signif(auc,digits = 2))),  cex=0.8, bty = "n")
  dev.off()
}

for (z in seq(length(utimes))){
  out_1=out[,,z]
  auc=c()
  for ( n in 2:nrow(out_1)){
    pred_= colMedians(out_1[1:n,], na.rm = TRUE)
    act=pvirus
    pred=prediction(pred_,act)
    perf <- performance(pred, "tpr", "fpr") 
    auc.perf = performance(pred, measure = "auc")
    auc=c(auc,auc.perf@y.values[[1]])
  }
  pdf(sprintf('%s/%sAUCvsiter_.pdf', outpath, utimes[z]))
  plot(auc, type ='l', xlab= 'Iterations', ylab= 'AUC')
  dev.off()
}


##outputing graph of important features in random forest models and their correlation
selmat=cbind( datamat[[1]], datamat[[2]],datamat[[3]], datamat[[4]] )
selnames=c(colnames(datamat[[1]]), colnames(datamat[[2]]),colnames(datamat[[3]]), colnames(datamat[[4]]))
nodelabs=rep(seq(length(datasets)), times=c(   ncol(datamat[[1]]), ncol(datamat[[2]]),ncol(datamat[[3]]), ncol(datamat[[4]]) ))
mycols=c('#EB9486', '#A3E7D6','#91C4F2', '#FCFF6C',  '#131611')


library(Hmisc)
library(igraph)
graphfeatures=selmat
pvs=matrix(NA, ncol(graphfeatures), ncol(graphfeatures))
for (i in seq(ncol(graphfeatures))){
    print(i)
    for (j in seq(ncol(graphfeatures)))
        pvs[i,j]=cor.test(graphfeatures[,i], graphfeatures[,j])$p.value
}


set.seed(2018)
library(Rtsne)
lo=Rtsne(pvs, perplexity=10, check_duplicates = FALSE)$Y
cols=mycols[c(nodelabs)]

for (z in seq(length(utimes))){
    selpvs=c(pvalues[[z]][[1]], pvalues[[z]][[2]], pvalues[[z]][[3]], pvalues[[z]][[4]])
    selimportance=c(scale(data.frame(importance[[z]][[1]])), scale(data.frame(importance[[z]][[2]])), scale(data.frame(importance[[z]][[3]])) , scale(importance[[z]][[4]]) )#, scale(data.frame(importance[[z]][[4]])), scale(data.frame(importance[[z]][[5]])))

    temp=cbind(selpvs, selimportance)
    colnames(temp)=c('pvalue', 'importance')
    rownames(temp)=selnames
    write.csv(temp, sprintf('%s/%s.csv', outpath, utimes[z]))

    pdf(sprintf('%s/%sbarplot.pdf', outpath, utimes[z]), width=10, height=7.5)
    barplot(-log10(cpvals[z,]), col=c(mycols, '#DC143C'), names.arg=cdatanames, ylim=c(0,5.5))
    abline(h=-log10(0.05), col=2, lwd=2, lty=2)
    dev.off()

    pdf(sprintf('%s/%scornetwork.pdf', outpath, utimes[z]), width=10*5, height=7.5*5)
    plot(lo, col=cols, pch=20, cex=(-log10(selpvs)), axes=FALSE, xlab='', ylab='')
    #text(lo, labels=selnames, cex=0.9, font=2)
    m=ncol(pvs)
    index=which(pvs< (0.000000000000000000005 / (m * (m-1) /2)), arr.ind=TRUE)
    #index=which(pvs< sort(pvs, decreasing=FALSE)[20000], arr.ind=TRUE)
    if (nrow(index)>0)
        segments(lo[index[,1],1], lo[index[,1],2], lo[index[,2],1], lo[index[,2],2], col='gray90', lwd=0.1*10)
    points(lo, col=cols, pch=20, cex=(-log10(selpvs))*5, xlab='', ylab='')
    text(jitter(lo), colnames(selmat), cex=2)
    legend('bottomleft', legend=c(datasets), inset=0.01, col=mycols, pch=20, pt.cex=2, cex=0.5)
    dev.off()

    pdf(sprintf('%s/%scornetworknoname.pdf', outpath, utimes[z]), width=10, height=7.5)
    plot(lo, col=cols, pch=20, cex=(-log10(selpvs)), axes=FALSE, xlab='', ylab='')
    m=ncol(pvs)
    index=which(pvs< (0.000000000000000000005 / (m * (m-1) /2)), arr.ind=TRUE)
    #index=which(pvs< sort(pvs, decreasing=FALSE)[20000], arr.ind=TRUE)
    if (nrow(index)>0)
        segments(lo[index[,1],1], lo[index[,1],2], lo[index[,2],1], lo[index[,2],2], col='gray90', lwd=0.1)
    points(lo, col=cols, pch=20, cex=(-log10(selpvs))*3, xlab='', ylab='')
    ##text(lo, colnames(selmat), cex=0.2)
    legend('bottomleft', legend=c(datasets), inset=0.01, col=mycols, pch=20, pt.cex=2, cex=0.5)
    dev.off()


}

