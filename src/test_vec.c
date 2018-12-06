#include <stdio.h>
#include <math.h>
#include "opus_types.h"
#include "arch.h"
#include "common.h"
#include "tansig_table.h"

#define LPCNET_TEST

#ifdef __ARM_NEON__
#define celt_exp2_neon celt_exp2_fast
#define tansig_approx_neon tansig_approx_fast
#define sigmoid_neon_approx sigmoid_approx_fast
#define softmax_neon softmax_fast
#define vec_tanh_neon vec_tanh_fast
#define vec_sigmoid_neon vec_sigmoid_fast
#define sgemv_accum16_neon sgemv_accum16_fast
#define sparse_sgemv_accum16_neon sparse_sgemv_accum16_fast
#include "vec_neon.h"
#endif

#include "vec.h"

#define ROW_STEP 16
#define ROWS     ROW_STEP*2
#define COLS     2
#define ENTRIES  2

int test_sgemv_accum16() {
  float weights[ROWS*COLS];
  float x[COLS];
  float out[ROWS], out_fast[ROWS];
  int i;

  printf("sgemv_accum16.....................: ");
  for(i=0; i<ROWS*COLS; i++) {
    weights[i] = i;
  }
  for(i=0; i<ROWS; i++) {
    out[i] = 0;
    out_fast[i] = 0;
  }
  
  for(i=0; i<COLS; i++) {
    x[i] = i+1;
  }

  sgemv_accum16(out, weights, ROWS, COLS, 1, x);
  sgemv_accum16_fast(out_fast, weights, ROWS, COLS, 1, x);

  for(i=0; i<ROWS; i++) {
    if (out[i] != out_fast[i]) {
      printf("fail\n");
      for(i=0; i<ROWS; i++) {
	printf("%d %f %f\n", i, out[i], out_fast[i]);
	if (out[i] != out_fast[i])
	  return 1;
      }
    }
  }

  printf("pass\n");
  return 0;
}


int test_sparse_sgemv_accum16() {
  int rows = ROW_STEP*ENTRIES;
  int indx[] = {1,0,2,0,1};
  float w[ROW_STEP*(1+2)];
  float x[ENTRIES] = {1,2};
  float out[ROW_STEP*(1+2)], out_fast[ROW_STEP*(1+2)];
  int i;

  printf("sparse_sgemv_accum16..............: ");
  for(i=0; i<ROW_STEP*(1+2); i++) {
    w[i] = i;
    out[i] = 0;
    out_fast[i] = 0;
  }
  
  sparse_sgemv_accum16(out, w, rows, indx, x);
  sparse_sgemv_accum16_fast(out_fast, w, rows, indx, x);

  for(i=0; i<ROW_STEP*ENTRIES; i++) {
    if (out[i] != out_fast[i]) {
      printf("fail\n");
      for(i=0; i<ROW_STEP*ENTRIES; i++) {
	printf("%d %f %f\n", i, out[i], out_fast[i]);
	if (out[i] != out_fast[i])
	  return 1;
      }
    }
  }

  printf("pass\n");
  return 0;
}

int main() {
  int test1 = test_sgemv_accum16();
  int test2 = test_sparse_sgemv_accum16();
  return test1 || test2;
}

  
