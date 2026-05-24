library(fru)

data(iris)
set.seed(17)

# Basic tests

fru(iris[,-5],iris[,5],trees=5000,importance=TRUE,forest=TRUE,oob=TRUE,threads=2)->model
imp<-importance(model)
op<-predict(model,threads=2)
pp<-predict(model,iris[,-5],threads=2)

stopifnot(mean(op==iris[,5])>0.9)
stopifnot(all(rank(imp)==c(2,1,4,3)))
stopifnot(mean(pp==iris[,5])>0.9)

print(model)

# Verify serialisation

reserialise<-function(x){
  cc<-rawConnection(raw(),"w+")
  serialize(x,connection=cc)
  stream<-rawConnectionValue(cc)
  ans<-unserialize(rawConnection(stream))
  ans
}

solidify(model)
reserialise(model)->model_copy

imp2<-importance(model_copy)
op2<-predict(model_copy,thread=2)
pp2<-predict(model_copy,iris[,-5],thread=2)

stopifnot(identical(imp,imp2))
stopifnot(identical(op,op2))
stopifnot(identical(pp,pp2))

# Run finalisation

rm(model)
gc()
gc()
gc()


# Run regression

fru(iris[,-2],iris[,2],trees=500,importance=TRUE,forest=TRUE,oob=TRUE,threads=2)->model
predict(model,iris[,5:1])->pred
stopifnot(cor(pred,iris[,2])>0.9)

# Check unwinding

poisoned_iris<-iris
poisoned_iris[,3]<-poisoned_iris[,3]*(1+2i)

try(fru(poisoned_iris[,-2],poisoned_iris[,2],threads=2))->exp_err
stopifnot(inherits(exp_err,"try-error"))


