#' @useDynLib fru, .registration=TRUE
NULL

#' Train the fru model
#'
#' Fru is an implementation of Leo Breiman's Random Forest (tm) method.
#' It fits an ensemble of decision trees built on bootstrap resamples of observations and additionally permuted by constraining split optimisation to a random subset of features.
#' The ensemble prediction is than established from individual trees by voting.
#' Thanks to its construction, the model can also provide a cross-validation-like internal approximation of error, so called out-of-bag predictions, as well as importance scores for features.
#'
#' In comparison to similar packages, fru is a tailored towards stability, correctness, efficiency and scalability on modern multithreaded machines, providing solid foundation for large data analysis, higher-level methods or production pipeline.
#' To this end, fru exposes only the original hyper-parameters and provides only the permutational importance, though calculated with a novel algorithm that alleviates its greater computational burden.
#'
#' Fru accepts logical, numeric (including integer) and factor features; NAs are not allowed and will result in error.
#' Logical features are always split into false/true groups without optimisation, yet are scored via weighted Gini impurity (for classification) or variance reduction (for regression), in order to be compared with splits on other features.
#' For numerical features, threshold value is optimised by an exhaustive scan of the criterion above; real values get threshold as a mid-point between values around the split, while integer values as a minimal of the two.
#' In case of a tie in the score, a smaller threshold is used.
#' Ordered factors or factors with six or more levels are treated as numerical, so follow the above procedure.
#' Unordered factors with five or less levels are split by finding a level partition into two subsets via an exhaustive scan of all possibilities, scored, as above, by Gini impurity or variance drop, depending on the forest type.
#' 
#' The maximal tree depth is hard-coded to 512; a critical sample size that triggers branch termination into leaf is one for classification and four for regression; this means that regression needs at least ten objects to be practical.
#' Leaves may be formed from larger samples in same cases, for instance when no split can be found based on the feature samples; in this case, for classification, random tie breaking is used.
#'
#' Fru uses its own PRNG, the pcg32 method by Melissa E. O'Neill, for its capacity to produce reasonably decorrelated streams, which are used to provide reproducibility of the output in parallel scenarios, regardless of the number of threads.
#' Namely, fru guarantees that the same trees will be fit for the same input and random seed, although their order may differ.
#' Thus, OOB predictions and importance scores will be the same up to numerical errors.
#' PRNG is used in training and prediction on new data; generator is seeded from the R generator, thus standard R interface of \code{set.seed} should be used to control it.
#' @param x Data frame containing predictors; must only contain logical, numeric, integer or factor columns, without NAs.
#' @param y Decision; either a factor or logical, for classification, or numeric, for regression.
#'  Integer decision will be silently converted into a real vector and treated as such afterwards.
#'  NA values are not accepted.
#' @param trees Number of trees to grow, a single number larger than zero.
#'  Also called \code{ntree} in other software.
#'  500 by default, in principle should be set to be big enough to stabilise the outputs of the forest, either prediction accuracy or importance; generally, bigger sets will need more trees, and it is unlikely that overshooting ensemble size will hurt the model in a statistically significant way.
#' @param tries Number of features to try at each split, a single number larger than zero and not larger than the number of columns in \code{x}.
#'  Also called \code{mtry} in other software.
#'  By default, set to the rounded square root of the number of features.
#'  It is unlikely this needs tweaking; increasing this value leads to a more accurate decision trees, but in turn makes them more correlated, spoiling the ensemble effect.
#' @param importance If set to \code{TRUE}, importance scores will be calculated.
#' @param oob If set to \code{TRUE}, out-of-bag (OOB) predictions will be calculated.
#' @param forest If set to \code{TRUE}, the forest object is returned and can be used for prediction.
#' @param solidify If set to \code{TRUE}, the forest object will use more memory but will survive serialisation, in particular when saved
#'  by \code{save}, \code{saveRDS} or when sent between processes.
#'  This can be done later with \code{solidify}, unless the model structure was already lost.
#' @param threads Number of threads to use; by default, or when set to 0, fru will try to use all available computing cores.
#' @returns The fitted model, an object of a class \code{fru}.
#' @references
#' Breiman L. (2001). \emph{Random Forests}, Machine Learning 45, 5-32.
#' @references 
#' O'Neil Melissa E. (2014). \emph{PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation}, HMC-CS-2014-0905.
#' @examples
#' set.seed(1)
#' data(iris)
#' fru(iris[,-5],iris[,5],threads=2)
#' @export
fru<-function(
  x,y,
  trees=500L,tries,
  forest=FALSE,oob=TRUE,importance=FALSE,solidify=FALSE,
  threads=0L){

  if(missing(tries)) tries<-round(sqrt(ncol(x)))
  if(forest) stopifnot(all(!duplicated(names(x))))
  
  ans<-.Call(C_fru,x,y,as.integer(trees),as.integer(tries),importance,oob,forest,as.integer(threads))
  # C code will validate flags
  if(oob || forest) ans$y<-y
  if(forest){
    # Store data structure for prediction
    ans$xn<-names(x)
    which(sapply(x,is.factor))->fax
    if(length(fax)>0)
      ans$fl<-stats::setNames(
        lapply(fax,function(e) levels(x[,e])),
        names(x)[fax])
  }
  if(importance) ans$xn<-names(x)
  class(ans)<-"fru"
  if(solidify) solidify(ans)
  ans
}

#' Solidify a given fru object
#'
#' Forces a model to be solidified, so that it would survive through saving into RDS or sending over a network, etc.
#' The downside is that the forest (and/or OOB scores or importance) will exist twice in the memory, and this process takes some time.
#' The function converts the object in-place, thanks to the semantics of external pointers.
#' No-op when given an object that is already serialised, either by \code{solidify=TRUE} flag passed to \code{fru}, due to a previous call to \code{solidify} or when deserialised.
#' @param x The fru model object.
#' @returns Invisibly, the same object as \code{x}; the function is called for the side effect of modifying the object.
#' @export
solidify<-function(x){
  stopifnot(inherits(x,"fru"))

  .Call(C_solidify,x$model)
  invisible(x)
}

#' Print the fru object
#'
#' Prints the basic information about the fitted model, in particular the OOB error estimated (if enabled previously).
#' @method print fru
#' @param x Model to print.
#' @param ... Ignored.
#' @returns Invisibly, the same object \code{x}.
#' @export
print.fru<-function(x,...){
  stopifnot(inherits(x,"fru"))
  nfo<-fru_info(x)
  if(nfo[1]==-1){
    cat("\n Empty fru model\n\n")
    return(invisible(x))
  }
  cat("\n Fru forest of",nfo[1],"trees\n\n")
  if(nfo[3]>0){
    oob<-predict.fru(x)
    if(is.factor(x$y)){
      acc<-mean(x$y==oob,na.rm=TRUE)
      print(table(True=x$y,OOB=oob,useNA="ifany"))
      cat(sprintf(" OOB error %0.2f%%\n\n",100*(1-acc)))
    }else if(is.numeric(x$y)){
      mse<-mean((x$y-oob)^2,na.rm=TRUE)
      cat(sprintf(" OOB MSE %0.2f, %%Var %0.2f%%\n\n",mse,100*(1-mse/stats::var(x$y))))
    }else if(is.logical(x$y)){
      acc<-mean(x$y==oob,na.rm=TRUE)
      print(table(True=x$y,OOB=oob,useNA="ifany"))
      cat(sprintf(" OOB error %0.2f%%\n\n",100*(1-acc)))
    }
  }
  if(nfo[2]>0) cat(" Forest available\n")
  if(nfo[4]>0) cat(" Importance available\n")
  cat("\n")
  invisible(x)
}

#' Predict with the fru model
#'
#' Either predicts a given new data or returns the OOB predictions of the model; optionally, for classification forests, returns raw votes for each decision class.
#'
#' If given, new data has to hold the same features as the training data, and the method will match them by name (order is irrelevant, additional features will be ignored); matched features have to be of the same type.
#' Moreover, factor features have to have exactly the same levels in the same order as in training; this will be checked.
#' 
#' The voting in classification case may lead to ties, in which case predict will use PRNG to resolve them.
#' In the OOB mode, the constant seed is used, so that OOB scores for the same forest model will always be the same, mimicking the behaviour of other packages which usually calculate predictions during training and store them with ties resolved at that time.
#' For new data prediction, PRNG is seeded from R's random state, so, in principle, ties will be resolved differently on each prediction.
#' If determinism is desired, it is best to use the votes output in which ties are evident.
#' Regression is performed using leaf averages, which is a deterministic process (not counting numerical issues possibly caused by nondeterministic order in which trees are produced when using multi-threading).
#'
#' The OOB predictions may contain NAs when a given object was not an OOB object of any tree, which may happen for small ensembles (in particular surely when \code{trees=1}).
#' Similarly, the sums of OOB votes for each object will not sum up to the ensemble size, but will for new data prediction.
#' 
#' By the nature of the method, new data prediction for the training data is usually close to perfect reproduction of the training decision; it is basically useless for any practical use.
#'
#' This method checks matches the input structure with the training data structure retained in the object, which may take some time, especially when data is large or short prediction latency is required.
#' In that case, one may use the non-exported \code{unsafe_fru_predict} function which expects \code{x} to be exactly in the same form as training, jumps straight to the compiled code and returns the predictions in the raw form (classes are level indices, vote matrix is unrolled, etc.).
#' @method predict fru
#' @param object A model used for prediction; has to hold the forest (\code{forest=TRUE} flag passed to \code{fru}) to make predictions on new data, or has to have OOB scores (\code{oob=TRUE} flag passed to \code{fru}) to return OOB scores.
#' @param x Data frame to predict; if missing or NULL, the method will return OOB scores.
#' @param votes If set to \code{TRUE}, changes the output to sums of votes cast by the ensemble on each class; useful as a prediction confidence score, for instance for ROC analysis.
#'  Only makes sense for classification; passing this flag together with regression forest will throw an error.
#' @param threads Number of threads to use; by default, or when set to 0, fru will try to use all available computing cores.
#' @param ... Ignored.
#' @examples
#' set.seed(1)
#' data(iris)
#' iris[c(TRUE,FALSE),]->iris_train
#' iris[c(FALSE,TRUE),]->iris_test
#' fru(iris_train[,-5],iris_train[,5],threads=2,forest=TRUE)->model
#' print(model)
#' table(predict(model,iris_test,threads=2),iris_test$Species)
#' @returns For a default of \code{votes=FALSE}, a vector with a prediction for either each row of \code{x}, or, when not given, an OOB approximated prediction for each row of the original training data.
#'  For \code{votes=TRUE}, a data frame with as many columns as decision classes, rows corresponding to rows of \code{x} or training data, and cells with the counts of votes per each class.
#' @export
predict.fru<-function(object,x,votes=FALSE,threads=0L,...){
  stopifnot(inherits(object,"fru"))
  if(missing(x)) x<-NULL
  if(!is.null(x)){
    if(is.null(object$xn)) stop("No forest in the fru object")
    x<-x[,object$xn]
    if(!is.null(x$fl))
      for(e in names(x$fl))
        if(!identical(x$fl[[e]],levels(x[,e])))
          stop("Levels mismatch in feature ",e)
  }
  ans<-.Call(C_predict,object$model,x,votes,threads)
  if(is.factor(object$y)){
    if(!votes){
      class(ans)<-"factor"
      levels(ans)<-levels(object$y)
    }else{
      l<-levels(object$y)
      ans<-data.frame(matrix(ans,ncol=length(l)))
      rownames(ans)<-rownames(x)
      names(ans)<-l
    }
  }else if(is.logical(object$y)){
    if(!votes){
      ans<-ans==2
    }else{
      ans<-data.frame(matrix(ans,ncol=2))
      rownames(ans)<-rownames(x)
      names(ans)<-c("FALSE","TRUE")
    }
  }
  ans
}

unsafe_fru_predict<-function(object,x,votes=FALSE,threads=0L){
  .Call(C_predict,object$model,x,votes,threads)
}

#' @rdname importance
#' @export
importance<-function(x,...)
  UseMethod("importance")

#' Extract importance
#'
#' Extracts importance from the \code{fru} model.
#' @note Other packages often scale importance by its standard error estimate, thus producing scales importance values square root of tree count times larger than fru.
#' If you get a "non applicable method" error, this method was probably shadowed by other package.
#' Use \code{fru:::importance} to call this function explicitly.
#' @rdname importance
#' @method importance fru
#' @param x A model from which importance scores should be extracted; has to hold importance scores (\code{importance=TRUE} flag passed to \code{fru}).
#' @param scale If \code{TRUE}, importance scores will be scaled their standard deviation over the ensemble.
#' @param ... Ignored.
#' @examples
#' set.seed(1)
#' data(iris)
#' fru(iris[,-5],iris[,5],threads=2,importance=TRUE)->model
#' importance(model)
#' @returns A vector of importance scores, in the order and named as columns were in the training data.
#' @export
importance.fru<-function(x,scale=FALSE,...){
  stopifnot(inherits(x,"fru"))
  stats::setNames(.Call(C_importance,x$model,scale),x$xn)
}

fru_info<-function(x){
  stopifnot(inherits(x,"fru"))
  # ntree, trees, oob, importance, ncls, n, m
  .Call(C_info,x$model)
}

#' Extract the forest
#'
#' Extracts the whole decision forest as a left-first, depth-first walk over all vertices.
#' @param x A model to convert; has to hold the forest (\code{forest=TRUE} flag passed to \code{fru}).
#' @returns A data frame with the forest structure.
#' Each row represents a step in a left-first, depth-first walk over the forest.
#' The \code{Feature} column holds, for branches, the feature used for a split, or NA, for leaves.
#' Similarly, \code{Threshold} and \code{Subset} columns hold the splitting criterion for branches; they exists only when holding any data.
#' For a numerical or integer split, observations with values strictly larger than threshold are sent left.
#' For a subset split, observations with values in the threshold subsets are sent left.
#' Logical splits have a fixed criterion, \code{TRUE}s are sent left.
#' This way, they have no corresponding criterion column.
#' Finally, leaf visits have their vote stored in the \code{Vote colum}.
#' @note This function will solidify the model object.
#' @export
extract_forest<-function(x){
  .Call(C_extract_forest,solidify(x)$model)->ans
  if(is.null(ans[[2]])) stop("Forest was not saved")
  ans[2:4]->ans
  data.frame(ans)->ans
  names(ans)<-c("Flag","Feature","Value")
  flags<-c("Leaf","Leaf","Logical","Real","Integer","Subset")
  ans$Flag<-factor(flags[ans$Flag])
  ans$Feature<-factor(x$xn[ans$Feature+1])
  ans$Threshold<-ifelse(ans$Flag%in%c("Real","Integer"),ans$Value,NA)
  if(all(is.na(ans$Threshold))) ans$Threshold<-NULL
  if(any(ans$Flag=="Subset")){
    Subsets<-rep(list(NA),nrow(ans))
    for(e in which(ans$Flag=="Subset")){
      mask<-which(intToBits(ans$Value[e])>0)
      subset<-as.character(x$fl[[as.character(ans$Feature[e])]][mask])
      Subsets[[e]]<-subset
    }
    ans$Subsets<-Subsets
  }
  ans$Vote<-ifelse(ans$Flag=="Leaf",ans$Value,NA)
  if(is.factor(x$y)){
    ans$Vote<-levels(x$y)[ans$Vote+1]
  }else if(is.logical(x$y)){
    ans$Vote<-ans$Vote!=0
  }
  ans$Flag<-NULL
  ans$Value<-NULL
  ans
}
