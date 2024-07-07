##Model to predict susceptibility to virus shedding prior to challenge
##The model was built using data from cytokine/chemokines, CBC, cell abundance, and cell activation and proliferation (median CD38 and Ki-67 expression)

##define datapaths
datapath='/Users/zainabrahil/Desktop/Zainab WCCTG14/ElasticNet'
outpath='/Users/zainabrahil/Desktop/Zainab WCCTG14/ElasticNet/JCI/MAY072020/Predictor/1_1_split'

##CSV files to read --> put into a dataset array containing each csv file
datasets=c( 'Luminex_3'  , 'CBC_3', 'Cell Quantity Reordered_06MAY2020', 'Medians_06MAY2020-colremoved' )
datanames=c('Luminex', 'CBC', 'Cell Quantity', 'Medians')##Name these data cells

##Name these data cells
outcomes=read.csv(sprintf('%s/%s.csv', datapath, 'Outcomes binary_11JUN2018'), header=TRUE)
##hai-defining shedders from non-shedders
hai=outcomes[,3]
##initializng index
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

utimes=c(-1,1:8, 29, 60)
upatients=unique(patients)

##added for addtional Y
Y_sub=vector()

##train time enter the days on which you want the model to train (set to day 6)
##iitializing data matrix-re-aaranging data
time_train=c(7)
for(j in seq(length(time_train))){
  for (i in seq(upatients)){
    
    Y_sub[length(upatients)*(j-1)+i]= freqs[ (patients== upatients[i] &  times==time_train[j] ),3]
  }
}


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
library(ROCR)
library(xgboost)

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
comb_pvals=matrix(NA,length(utimes), 5)
ansmat_list=list()

##running analysis for day 1 (index 2)
for( z in  2){
  print(z)
  importance[[z]]=list()
  pvalues[[z]]=list()
  z_test=utimes[z]  
  index=vector()
  index_test=vector()
  
  
  for (i in seq(length(upatients))){
    index_test[i]=which(rownames(datamat[[1]])==sprintf('%s_%d',upatients[i], z_test))
    }
  
  for(j in seq(length(time_train))){
    for (i in seq(length(upatients))){
      index[length(upatients)*(j-1)+i]=which(rownames(datamat[[1]])==sprintf('%s_%d',upatients[i], utimes[time_train[j]]))
    }
  }
  
  pdatamat=list()
  pdatamat_test=list()
  
  
  for (i in seq(length(datamat))){
    pdatamat[[i]]=datamat[[i]][index,]
    pdatamat_test[[i]]=datamat[[i]][index_test,]
  }
  #iterations set to 100
  ansmat_list[[z]]=array(NA,c(100, length(upatients), length(pdatamat)))  
  for(d in seq(length(pdatamat))){        
    print(paste("you are processing dataset number", d))
    X=pdatamat[[d]]
    X_test=pdatamat_test[[d]]
    Y=factor(pvirus)
    Y= ifelse(pvirus == 'TRUE',1,0)
    Y=Y_sub
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
    
    #iter
    ansmat=matrix(NA, 100, nrow(X_test))
    imps=rep(0, ncol(X))
    
    ##pvs for the test dataset
    pvs=vector()
    for (iter in seq(ncol(X))){
      pvs[iter]=wilcox.test(X_test[,iter]~factor(pvirus))$p.value
    } 
    
    for (iter in seq(nrow(ansmat))){
      
      train_id=sample(seq(length(upatients)),length(upatients)*1/2)
      train_day_max= length(Y)/length(upatients)
      train =c()
      for (days in seq(1:train_day_max)){
        t= train_id+((days-1)*length(upatients))
        train= c(train, t)
      }
      if (min(table(Y[train]))==0)
        next
      test=seq(length(upatients))[-train_id]
      mod=randomForest(X[train,], Y[train], na.rm=TRUE)
      top40_X=which((importance(mod)>0.09))
      
      dtrain <- xgb.DMatrix(data = X_test[train_id,top40_X], label= Y[train_id])
      negative_cases <- sum(Y[train_id] == 0)
      postive_cases <- sum(Y[train_id] == 1)
      xgb <- xgboost(data = dtrain,  max.depth = 3,  nround = 50, early_stopping_rounds = 3,scale_pos_weight = negative_cases/postive_cases,gamma=1,verbose =0, objective = "binary:logistic")
      ansmat[iter, test]=  predict(xgb, X[test,top40_X])
      mod_new=randomForest(X_test[train_id,top40_X], Y[train_id], na.rm=TRUE)
      imps[top40_X]=imps[top40_X]+importance(mod_new)
       #imps=imps+importance(mod)
    }
    imps=imps/nrow(ansmat)
    
    importance[[z]][[d]]=imps
    pvalues[[z]][[d]]=pvs
    ansmat_list[[z]][,,d]=ansmat
    ans=rowMedians(t(ansmat))
    
    modelpvals[z,d]=wilcox.test(ans ~ pvirus)$p.value
    preds[z,d,(index_test)-35*(z-1)]=ans
  }
}


###initializing vector for combined p-values
combinedpvals=vector()
for (z in 2){
  ans=rowMedians(t(preds[z,,]))
  if(length(table(ans))==0){
    combinedpvals[z]=1
    next
  }
  combinedpvals[z]=wilcox.test(ans ~ pvirus)$p.value
}

ansmat_list_combine=array(NA,c(nrow(ansmat), length(upatients), length(utimes)))
for( z in 2){
  for(r in seq(nrow(ansmat))){
    ansmat_list_combine[,,z][r,]= rowMedians(ansmat_list[[z]][r,,])
  }
}




##outputing ROC for day 1
out=ansmat_list_combine
library(matrixStats)
for( d in 2){
  day=d
  out_1=out[,,d]
  pred_= colMedians(out_1, na.rm = TRUE)
  pred_=pred_
  act=pvirus
  save_name=paste0('%s/ROC_400iter',utimes[day], '.pdf')
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

for (z in 2){
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


library(gplots)
library(RColorBrewer)
cpvals=cbind(modelpvals, combinedpvals)
cdatanames=c(datanames, 'Combined')
temp=matrix(p.adjust(cpvals, method = 'fdr'), nrow=nrow(modelpvals))
Colors=(brewer.pal(8,"Blues"))
Colors=colorRampPalette(Colors)(120)
heatmap.2(t(-log10(temp)), col=Colors, ,Colv=NA, Rowv=NA, scale='none', trace='none', labRow=cdatanames, labCol=utimes, margins=c(5, 10), key.xlab='-log10(pvalue)', density.info = 'none')
dev.off()
colnames(temp)=cdatanames
rownames(temp)=utimes
#storing pvalues as a csv for later  reference
write.csv(temp, sprintf('%s/pvalues_heatmap.csv', outpath))


selmat=cbind( datamat[[1]], datamat[[2]],datamat[[3]], datamat[[4]] )
selnames=c(colnames(datamat[[1]]), colnames(datamat[[2]]),colnames(datamat[[3]]), colnames(datamat[[4]]))
nodelabs=rep(seq(length(datasets)), times=c(   ncol(datamat[[1]]), ncol(datamat[[2]]),ncol(datamat[[3]]), ncol(datamat[[4]]) ))
mycols=c('#EB9486', '#A3E7D6','#91C4F2', '#FCFF6C',  '#131611')

##outputing graph of important features in random forest models and their correlation
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
lo=Rtsne(pvs, perplexity=10,  check_duplicates = FALSE)$Y
cols=mycols[c(nodelabs)]

for (z in 2){
  selpvs=c(pvalues[[z]][[1]], pvalues[[z]][[2]], pvalues[[z]][[3]], pvalues[[z]][[4]])
  selimportance=c(scale(data.frame(importance[[z]][[1]])), scale(data.frame(importance[[z]][[2]])), scale(data.frame(importance[[z]][[3]])) , scale(importance[[z]][[4]]) )#, scale(data.frame(importance[[z]][[4]])), scale(data.frame(importance[[z]][[5]])))
  
  temp=cbind(selpvs, selimportance)
  colnames(temp)=c('pvalue', 'importance')
  rownames(temp)=selnames
  write.csv(temp, sprintf('%s/%s.csv', outpath, utimes[z]))
  
  pdf(sprintf('%s/%sbarplot.pdf', outpath, utimes[z]), width=10, height=7.5)
  temp=matrix(p.adjust(cpvals, method = 'fdr'), nrow=nrow(modelpvals))
  barplot(-log10(temp[z,]), col=c(mycols, '#DC143C'), names.arg=cdatanames, ylim=c(0,5.5))
  abline(h=-log10(0.05), col=2, lwd=2, lty=2)
  dev.off()
  
  pdf(sprintf('%s/%scornetwork.pdf', outpath, utimes[z]), width=10*5, height=7.5*5)
  plot(lo, col=cols, pch=20, cex=(-log10(selpvs)), axes=FALSE, xlab='', ylab='')
  m=ncol(pvs)
  index=which(pvs< (0.000000000000000000005 / (m * (m-1) /2)), arr.ind=TRUE)
  if (nrow(index)>0)
    segments(lo[index[,1],1], lo[index[,1],2], lo[index[,2],1], lo[index[,2],2], col='gray90', lwd=0.1*10)
  points(lo, col=cols, pch=20, cex=(-log10(selpvs))*5, xlab='', ylab='')
  text(jitter(lo), colnames(selmat), cex=2)
  legend('bottomleft', legend=c(datasets), inset=0.01, col=mycols, pch=20, pt.cex=2, cex=0.5)
  dev.off()
  
  pdf(sprintf('%s/%scornetwork-condensed.pdf', outpath, utimes[z]), width=10, height=7.5)
  plot(lo, col=cols, pch=20, cex=(-log10(selpvs)), axes=FALSE, xlab='', ylab='')
  m=ncol(pvs)
  index=which(pvs< (0.000000000000000000005 / (m * (m-1) /2)), arr.ind=TRUE)
  #index=which(pvs< sort(pvs, decreasing=FALSE)[20000], arr.ind=TRUE)
  if (nrow(index)>0)
    segments(lo[index[,1],1], lo[index[,1],2], lo[index[,2],1], lo[index[,2],2], col='gray90', lwd=0.1)
  points(lo, col=cols, pch=20, cex=(-log10(selpvs))*1.4, xlab='', ylab='')
  ##text(lo, colnames(selmat), cex=0.2)
  legend('bottomleft', legend=c(datasets), inset=0.01, col=mycols, pch=20, pt.cex=2, cex=0.5)
  dev.off()
  
  
}

