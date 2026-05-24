#include <stdint.h>
#include <stddef.h>

#if __STDC_VERSION__ < 202311l
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void xrf_cls(int n,int m,int ntree,int mtry,void *att,
             int *y,int yc,int threads,
             void **forest,uint8_t todo,uint64_t seed);

void xrf_reg(int n,int m,int ntree,int mtry,void *att,
             double *y,int threads,
             void **forest,uint8_t todo,uint64_t seed);

void* importance_xrf(void *forest,double *importance,bool normalised,int *noimp);

void* predict_xrf(void *forest,void *att,int ncat,int n,int m,bool get_votes,
                  uint64_t seed,int threads,int *cat,double *num,bool *fail);

void finalise_xrf(void *forest);

void* info_xrf(void *forest,int *info);

void* codec_xrf(void *forest,void *ans);

#ifdef __cplusplus
}
#endif
