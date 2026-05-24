#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/Visibility.h>

// Import C headers for rust API
#include "fru/api.h"

void finalise_forest(SEXP sModel){
  void *model=R_ExternalPtrAddr(sModel);
  if(model) finalise_xrf(model);
}

void* try_deserialise(SEXP sModel){
  void *model=R_ExternalPtrAddr(sModel);
  if(model) return(model);
  
  //Rethink that
  SEXP sTag=R_ExternalPtrTag(sModel);
  if(Rf_length(sTag)!=3) Rf_error("Wrong model pointer");

  SEXP sSerModel=R_ExternalPtrProtected(sModel);
  //If sSerModel is NULL, it means the model was not solidified
  if(Rf_length(sSerModel)==0) return(NULL); 
  if(Rf_length(sSerModel)!=6) Rf_error("Corrupted model pointer");
  model=codec_xrf(NULL,sSerModel);

  R_SetExternalPtrAddr(sModel,model);
  return(model);
}

SEXP C_predict(SEXP sModel,SEXP sAttributes,SEXP sVotes,SEXP sThreads){
  int mp=Rf_length(sAttributes);
  
  void *model=try_deserialise(sModel);
  if(!model) Rf_error("Model is not available");
  SEXP sTag=R_ExternalPtrTag(sModel);
  int *tag=INTEGER(sTag);
  if(Rf_length(sTag)!=3) Rf_error("Wrong model pointer");
  int v[4];
  model=info_xrf(model, v);
  R_SetExternalPtrAddr(sModel,model);
  int has_trees=v[1]!=0;
  int has_oob=v[2]!=0;
  int threads=Rf_asInteger(sThreads);
  int get_votes=Rf_asLogical(sVotes);
  threads=(threads<0)?0:threads;

  if(mp>0 && !has_trees) Rf_error("Forest is required for prediction");
  if(mp==0 && !has_oob) Rf_error("OOB predictions were not produced");
  if(get_votes && tag[0]==0) Rf_error("Votes make no sense for regression");
  
  uint64_t seed=0;
  if(!get_votes && tag[0]>0 && mp>0){
    GetRNGstate(); 
    seed=(uint32_t)(((double)(~((uint32_t)0)))*unif_rand()); 
    uint64_t seed2=(uint32_t)(((double)(~((uint32_t)0)))*unif_rand()); 
    PutRNGstate(); 
    seed=(seed<<32)+seed2;
  }

  SEXP sPrediction;
  bool fail=false;
  int *ip=NULL;
  double *rp=NULL;
  int n_ans;
  if(mp==0){
    //OOB
    n_ans=tag[1];
  }else{
    n_ans=Rf_length(VECTOR_ELT(sAttributes,0));
  }
  if(tag[0]==0){
    sPrediction=PROTECT(Rf_allocVector(REALSXP,n_ans));
    rp=REAL(sPrediction);
    for(int e=0;e<n_ans;e++) rp[e]=NA_REAL;
  }else{
    if(get_votes) n_ans=n_ans*tag[0];
    sPrediction=PROTECT(Rf_allocVector(INTSXP,n_ans));
    ip=INTEGER(sPrediction);
    for(int e=0;e<n_ans;e++) ip[e]=NA_INTEGER;
  }
  if(mp==0){
    //OOB
    model=predict_xrf(model,NULL,tag[0],tag[1],tag[2],get_votes,seed,0,ip,rp,&fail);
    R_SetExternalPtrAddr(sModel,model);
  }else{
    if(mp!=tag[2]){
      Rf_error("Attribute count mismatch");
    }
    int n=Rf_length(VECTOR_ELT(sAttributes,0));
    model=predict_xrf(model,sAttributes,tag[0],n,tag[2],get_votes,seed,threads,ip,rp,&fail);
    R_SetExternalPtrAddr(sModel,model);
  }
  if(fail){
    Rf_error("Computational kernel has panicked");
  }
  
  UNPROTECT(1);
  return(sPrediction);
}

SEXP C_importance(SEXP sModel,SEXP sNormalise){
  void *model=try_deserialise(sModel);
  if(!model) Rf_error("Model is not available");
  SEXP sTag=R_ExternalPtrTag(sModel);
  int *tag=INTEGER(sTag);
  if(Rf_length(sTag)!=3) Rf_error("Wrong model pointer");
  
  bool nrm=Rf_asBool(sNormalise);
  int noimp=0;
  SEXP sImportance=PROTECT(Rf_allocVector(REALSXP,tag[2]));
  double *imp=REAL(sImportance);
  for(int e=0;e<tag[2];e++) imp[e]=NA_REAL;
  model=importance_xrf(model,imp,nrm,&noimp);
  R_SetExternalPtrAddr(sModel,model);
  if(noimp) Rf_error("No importance stored in the model");
  
  UNPROTECT(1);
  return(sImportance);
}

SEXP C_fru(SEXP sAttributes,SEXP sDecision,SEXP sNtree,SEXP sMtry,SEXP sImportance,SEXP sOob,SEXP sForest,SEXP sThreads){
  //Check sAttributes to be a data frame?
  int m=Rf_length(sAttributes);
  if(m==0) Rf_error("Fru needs at least one argument");
  int n=Rf_length(VECTOR_ELT(sAttributes,0));
  if(n==0) Rf_error("No observations");
  if(Rf_length(sDecision)!=n){
    Rf_error("Decision length differes from attribute lengths");
  }

  bool make_importance=Rf_asBool(sImportance);
  bool make_oob=Rf_asBool(sOob);
  bool save_forest=Rf_asBool(sForest);
  uint8_t todo=(make_importance?8:0)+(make_oob?4:0)+(save_forest?2:0);

  int ntree=Rf_asInteger(sNtree);
  if(ntree<1){
    Rf_warning("Parameter trees too low, changed to 1");
    ntree=1;
  }
  int mtry=Rf_asInteger(sMtry);
  int threads=Rf_asInteger(sThreads);
  threads=(threads<0)?0:threads;
  if(mtry<1){
    Rf_warning("Parameter tries too low, changed to 1");
    mtry=1;
  }else if(mtry>m){
    Rf_warning("Paramter tries capped to the size of x");
    mtry=m;
  }

  void *forest=NULL;
  int classes=0;
  double *yreg=NULL;
  int *ycls=NULL;
  switch(TYPEOF(sDecision)){
    case REALSXP:
      yreg=REAL(sDecision);
      for(int e=0;e<n;e++)
        if(!R_FINITE(yreg[e])) Rf_error("Only finite values allowed in decision");
      break;
    case LGLSXP:;
      int *v=INTEGER(sDecision);
      classes=2;
      ycls=(int*)R_alloc(sizeof(int),n);
      for(int e=0;e<n;e++){
        if(ycls[e]==NA_LOGICAL) Rf_error("NAs not allowed in decision");
        ycls[e]=v[e]?2:1;
      }
      break;
    case INTSXP:;
      int nc=Rf_length(Rf_getAttrib(sDecision,R_LevelsSymbol));
      if(Rf_isOrdered(sDecision)) nc=0;
      if(nc==0){
        //For the decision, we will treat integer stuff as real
        yreg=(double*)R_alloc(sizeof(double),n);
        int *yri=INTEGER(sDecision);
        for(int e=0;e<n;e++){
          if(yri[e]==NA_INTEGER) Rf_error("NAs not allowed in decision");
          yreg[e]=(double)yri[e];
        }
      }else{
        ycls=INTEGER(sDecision);
        for(int e=0;e<n;e++){
          if(ycls[e]==NA_INTEGER) Rf_error("NAs not allowed in decision");
        }
        classes=nc;
      }
      break;
    default:
      Rf_error("Unknown decision kind");
  }
  
   //Point of no return for jumping into Rust; RNG seeding
   GetRNGstate(); 
   uint64_t seed=(uint32_t)(((double)(~((uint32_t)0)))*unif_rand()); 
   uint64_t seed2=(uint32_t)(((double)(~((uint32_t)0)))*unif_rand()); 
   PutRNGstate(); 
   seed=(seed<<32)+seed2; 
   
  if(ycls){
    xrf_cls(n,m,ntree,mtry,(void*)sAttributes,ycls,classes,threads,&forest,todo,seed);
  }
  if(yreg){
    xrf_reg(n,m,ntree,mtry,(void*)sAttributes,yreg,threads,&forest,todo,seed);
  }
  if(!forest) Rf_error("Computational kernel of fru has thrown a panic; input data was likely invalid");
  

  SEXP sTag=PROTECT(Rf_allocVector(INTSXP,3));
  int *tag=INTEGER(sTag);
  tag[0]=classes; //Zero means regression
  tag[1]=n;
  tag[2]=m;
  SEXP sFo=PROTECT(R_MakeExternalPtr(forest,sTag,R_NilValue));
  R_RegisterCFinalizer(sFo,finalise_forest);

  
  SEXP sAns=PROTECT(Rf_allocVector(VECSXP,1)); // model, oob-err, oob preds, imp
  SET_VECTOR_ELT(sAns,0,sFo);
  SEXP sAnsNames=PROTECT(NEW_CHARACTER(1));
  SET_STRING_ELT(sAnsNames,0,Rf_mkChar("model"));
  Rf_setAttrib(sAns,R_NamesSymbol,sAnsNames);
  UNPROTECT(4);
  return sAns;
}

SEXP C_info(SEXP sModel){
  void *model=try_deserialise(sModel);
  SEXP sTag=R_ExternalPtrTag(sModel);
  int *tag=INTEGER(sTag);
  if(Rf_length(sTag)!=3) Rf_error("Wrong model pointer");
  
  // ntree, trees, oob, importnace, ncls, n, m
  SEXP sAns=PROTECT(Rf_allocVector(INTSXP,7));
  int *v=INTEGER(sAns);
  if(model){
    void *new_ptr=info_xrf(model, v);
    R_SetExternalPtrAddr(sModel,new_ptr);
  }else{
    v[0]=-1;
    v[1]=0;
    v[2]=0;
    v[3]=0;
  }
  v[4]=tag[0];
  v[5]=tag[1];
  v[6]=tag[2];
  
  UNPROTECT(1);
  return(sAns);
}

SEXP C_solidify(SEXP sModel){
  void *model=R_ExternalPtrAddr(sModel);
  if(!model) Rf_error("Model was already lost");
  SEXP sTag=R_ExternalPtrTag(sModel);
  if(Rf_length(sTag)!=3) Rf_error("Wrong model pointer");

  SEXP sAns=PROTECT(Rf_allocVector(VECSXP,6));
  model=codec_xrf(model,sAns);
  R_SetExternalPtrAddr(sModel,model);
  R_SetExternalPtrProtected(sModel,sAns);

  UNPROTECT(1);
  return(sAns);
}

SEXP C_extract_forest(SEXP sModel){
  void *model=try_deserialise(sModel);
  if(!model) Rf_error("Model is not available");
  R_SetExternalPtrAddr(sModel,model);
  return(R_ExternalPtrProtected(sModel));
}

#define CALLDEF(name,n) {#name,(DL_FUNC)&name,n}
static const R_CallMethodDef R_CallDef[]={
  CALLDEF(C_fru,8),
  CALLDEF(C_predict,4),
  CALLDEF(C_importance,2),
  CALLDEF(C_info,1),
  CALLDEF(C_solidify,1),
  CALLDEF(C_extract_forest,1),
  {NULL,NULL,0}
};

void attribute_visible R_init_fru(DllInfo *dll){
  R_registerRoutines(dll,NULL,R_CallDef,NULL,NULL);
  R_useDynamicSymbols(dll,FALSE);
  R_forceSymbols(dll,TRUE);
}

void pull_feature(SEXP sAttributes,int which,int *n,int *ti,void **data){
  SEXP sAtt=VECTOR_ELT(sAttributes,which);
  n[0]=Rf_length(sAtt);
  switch(TYPEOF(sAtt)){
    case LGLSXP:
      data[0]=(void*)INTEGER(sAtt);
      ti[0]=-2;
      break;
    case REALSXP:
      data[0]=(void*)REAL(sAtt);
      ti[0]=-1;
      break;
    case INTSXP:
      data[0]=(void*)INTEGER(sAtt);
      int32_t cat=Rf_length(Rf_getAttrib(sAtt,R_LevelsSymbol));
      ti[0]=cat;
      break;
    default:
      data[0]=NULL;
      ti[0]=-3;
      break;
  }
}

char* fill_raw_vec(SEXP sWithin,int index,int len){
  SEXP sAns=PROTECT(Rf_allocVector(RAWSXP,len));
  SET_VECTOR_ELT(sWithin,index,sAns);
  //From now on, PROTECTION will relay on within object
  char *ans=(char*)RAW(sAns);
  UNPROTECT(1);
  return(ans);
}

double* fill_double_vec(SEXP sWithin,int index,int len){
  SEXP sAns=PROTECT(Rf_allocVector(REALSXP,len));
  SET_VECTOR_ELT(sWithin,index,sAns);
  //From now on, PROTECTION will relay on within object
  double *ans=REAL(sAns);
  UNPROTECT(1);
  return(ans);
}

int* fill_int_vec(SEXP sWithin,int index,int len){
  SEXP sAns=PROTECT(Rf_allocVector(INTSXP,len));
  SET_VECTOR_ELT(sWithin,index,sAns);
  //From now on, PROTECTION will relay on within object
  int *ans=INTEGER(sAns);
  UNPROTECT(1);
  return(ans);
}

char* decode_raw_vec(SEXP sWithin,int index){
  return((char*)RAW(VECTOR_ELT(sWithin,index)));
}

int* decode_int_vec(SEXP sWithin,int index){
  return(INTEGER(VECTOR_ELT(sWithin,index)));
}

double* decode_double_vec(SEXP sWithin,int index){
  return(REAL(VECTOR_ELT(sWithin,index)));
}

int elem_length(SEXP sWithin,int index){
  return(Rf_length(VECTOR_ELT(sWithin,index)));
}

