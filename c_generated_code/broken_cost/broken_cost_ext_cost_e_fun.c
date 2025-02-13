/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) broken_cost_ext_cost_e_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};

static const casadi_real casadi_c0[121] = {1000., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1000., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2000., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 200., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 500., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1500., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 3.7500000000000000e+00};

/* broken_cost_ext_cost_e_fun:(i0[13],i1[],i2[],i3[17])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+2, *w2=w+13, *w3=w+16, *w4=w+19, w5, *w6=w+24, w7, w8, w9, w10, *w11=w+32, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, *w23=w+47, *w24=w+50, *w25=w+53, *w26=w+64, *w27=w+75;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(1x11) */
  casadi_clear(w1, 11);
  /* #2: @2 = input[0][0] */
  casadi_copy(arg[0], 3, w2);
  /* #3: @3 = input[3][0] */
  casadi_copy(arg[3], 3, w3);
  /* #4: @2 = (@2-@3) */
  for (i=0, rr=w2, cs=w3; i<3; ++i) (*rr++) -= (*cs++);
  /* #5: @4 = input[3][1] */
  casadi_copy(arg[3] ? arg[3]+3 : 0, 4, w4);
  /* #6: @5 = @4[0] */
  for (rr=(&w5), ss=w4+0; ss!=w4+1; ss+=1) *rr++ = *ss;
  /* #7: @6 = input[0][1] */
  casadi_copy(arg[0] ? arg[0]+3 : 0, 4, w6);
  /* #8: @7 = @6[0] */
  for (rr=(&w7), ss=w6+0; ss!=w6+1; ss+=1) *rr++ = *ss;
  /* #9: @8 = @6[1] */
  for (rr=(&w8), ss=w6+1; ss!=w6+2; ss+=1) *rr++ = *ss;
  /* #10: @8 = (-@8) */
  w8 = (- w8 );
  /* #11: @9 = @6[2] */
  for (rr=(&w9), ss=w6+2; ss!=w6+3; ss+=1) *rr++ = *ss;
  /* #12: @9 = (-@9) */
  w9 = (- w9 );
  /* #13: @10 = @6[3] */
  for (rr=(&w10), ss=w6+3; ss!=w6+4; ss+=1) *rr++ = *ss;
  /* #14: @10 = (-@10) */
  w10 = (- w10 );
  /* #15: @11 = vertcat(@7, @8, @9, @10) */
  rr=w11;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  /* #16: @7 = dot(@6, @6) */
  w7 = casadi_dot(4, w6, w6);
  /* #17: @11 = (@11/@7) */
  for (i=0, rr=w11; i<4; ++i) (*rr++) /= w7;
  /* #18: @7 = @11[0] */
  for (rr=(&w7), ss=w11+0; ss!=w11+1; ss+=1) *rr++ = *ss;
  /* #19: @8 = (@5*@7) */
  w8  = (w5*w7);
  /* #20: @9 = @4[1] */
  for (rr=(&w9), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #21: @10 = @11[1] */
  for (rr=(&w10), ss=w11+1; ss!=w11+2; ss+=1) *rr++ = *ss;
  /* #22: @12 = (@9*@10) */
  w12  = (w9*w10);
  /* #23: @8 = (@8-@12) */
  w8 -= w12;
  /* #24: @12 = @4[2] */
  for (rr=(&w12), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #25: @13 = @11[2] */
  for (rr=(&w13), ss=w11+2; ss!=w11+3; ss+=1) *rr++ = *ss;
  /* #26: @14 = (@12*@13) */
  w14  = (w12*w13);
  /* #27: @8 = (@8-@14) */
  w8 -= w14;
  /* #28: @14 = @4[3] */
  for (rr=(&w14), ss=w4+3; ss!=w4+4; ss+=1) *rr++ = *ss;
  /* #29: @15 = @11[3] */
  for (rr=(&w15), ss=w11+3; ss!=w11+4; ss+=1) *rr++ = *ss;
  /* #30: @16 = (@14*@15) */
  w16  = (w14*w15);
  /* #31: @8 = (@8-@16) */
  w8 -= w16;
  /* #32: @16 = (@5*@15) */
  w16  = (w5*w15);
  /* #33: @17 = (@9*@13) */
  w17  = (w9*w13);
  /* #34: @16 = (@16+@17) */
  w16 += w17;
  /* #35: @17 = (@12*@10) */
  w17  = (w12*w10);
  /* #36: @16 = (@16-@17) */
  w16 -= w17;
  /* #37: @17 = (@14*@7) */
  w17  = (w14*w7);
  /* #38: @16 = (@16+@17) */
  w16 += w17;
  /* #39: @11 = vertcat(@8, @0, @0, @16) */
  rr=w11;
  *rr++ = w8;
  *rr++ = w0;
  *rr++ = w0;
  *rr++ = w16;
  /* #40: @17 = ||@11||_F */
  w17 = sqrt(casadi_dot(4, w11, w11));
  /* #41: @11 = (@11/@17) */
  for (i=0, rr=w11; i<4; ++i) (*rr++) /= w17;
  /* #42: @17 = @11[0] */
  for (rr=(&w17), ss=w11+0; ss!=w11+1; ss+=1) *rr++ = *ss;
  /* #43: @18 = @11[1] */
  for (rr=(&w18), ss=w11+1; ss!=w11+2; ss+=1) *rr++ = *ss;
  /* #44: @18 = (-@18) */
  w18 = (- w18 );
  /* #45: @19 = @11[2] */
  for (rr=(&w19), ss=w11+2; ss!=w11+3; ss+=1) *rr++ = *ss;
  /* #46: @19 = (-@19) */
  w19 = (- w19 );
  /* #47: @20 = @11[3] */
  for (rr=(&w20), ss=w11+3; ss!=w11+4; ss+=1) *rr++ = *ss;
  /* #48: @21 = (-@20) */
  w21 = (- w20 );
  /* #49: @4 = vertcat(@17, @18, @19, @21) */
  rr=w4;
  *rr++ = w17;
  *rr++ = w18;
  *rr++ = w19;
  *rr++ = w21;
  /* #50: @17 = dot(@11, @11) */
  w17 = casadi_dot(4, w11, w11);
  /* #51: @4 = (@4/@17) */
  for (i=0, rr=w4; i<4; ++i) (*rr++) /= w17;
  /* #52: @17 = @4[1] */
  for (rr=(&w17), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #53: @18 = (@8*@17) */
  w18  = (w8*w17);
  /* #54: @19 = (@5*@10) */
  w19  = (w5*w10);
  /* #55: @21 = (@9*@7) */
  w21  = (w9*w7);
  /* #56: @19 = (@19+@21) */
  w19 += w21;
  /* #57: @21 = (@12*@15) */
  w21  = (w12*w15);
  /* #58: @19 = (@19+@21) */
  w19 += w21;
  /* #59: @21 = (@14*@13) */
  w21  = (w14*w13);
  /* #60: @19 = (@19-@21) */
  w19 -= w21;
  /* #61: @21 = @4[0] */
  for (rr=(&w21), ss=w4+0; ss!=w4+1; ss+=1) *rr++ = *ss;
  /* #62: @22 = (@19*@21) */
  w22  = (w19*w21);
  /* #63: @18 = (@18+@22) */
  w18 += w22;
  /* #64: @5 = (@5*@13) */
  w5 *= w13;
  /* #65: @9 = (@9*@15) */
  w9 *= w15;
  /* #66: @5 = (@5-@9) */
  w5 -= w9;
  /* #67: @12 = (@12*@7) */
  w12 *= w7;
  /* #68: @5 = (@5+@12) */
  w5 += w12;
  /* #69: @14 = (@14*@10) */
  w14 *= w10;
  /* #70: @5 = (@5+@14) */
  w5 += w14;
  /* #71: @14 = @4[3] */
  for (rr=(&w14), ss=w4+3; ss!=w4+4; ss+=1) *rr++ = *ss;
  /* #72: @10 = (@5*@14) */
  w10  = (w5*w14);
  /* #73: @18 = (@18+@10) */
  w18 += w10;
  /* #74: @10 = @4[2] */
  for (rr=(&w10), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #75: @12 = (@16*@10) */
  w12  = (w16*w10);
  /* #76: @18 = (@18-@12) */
  w18 -= w12;
  /* #77: @18 = sq(@18) */
  w18 = casadi_sq( w18 );
  /* #78: @8 = (@8*@10) */
  w8 *= w10;
  /* #79: @19 = (@19*@14) */
  w19 *= w14;
  /* #80: @8 = (@8-@19) */
  w8 -= w19;
  /* #81: @5 = (@5*@21) */
  w5 *= w21;
  /* #82: @8 = (@8+@5) */
  w8 += w5;
  /* #83: @16 = (@16*@17) */
  w16 *= w17;
  /* #84: @8 = (@8+@16) */
  w8 += w16;
  /* #85: @8 = sq(@8) */
  w8 = casadi_sq( w8 );
  /* #86: @18 = (@18+@8) */
  w18 += w8;
  /* #87: @3 = input[0][2] */
  casadi_copy(arg[0] ? arg[0]+7 : 0, 3, w3);
  /* #88: @23 = input[3][2] */
  casadi_copy(arg[3] ? arg[3]+7 : 0, 3, w23);
  /* #89: @3 = (@3-@23) */
  for (i=0, rr=w3, cs=w23; i<3; ++i) (*rr++) -= (*cs++);
  /* #90: @23 = input[0][3] */
  casadi_copy(arg[0] ? arg[0]+10 : 0, 3, w23);
  /* #91: @24 = input[3][3] */
  casadi_copy(arg[3] ? arg[3]+10 : 0, 3, w24);
  /* #92: @23 = (@23-@24) */
  for (i=0, rr=w23, cs=w24; i<3; ++i) (*rr++) -= (*cs++);
  /* #93: @25 = vertcat(@2, @18, @20, @3, @23) */
  rr=w25;
  for (i=0, cs=w2; i<3; ++i) *rr++ = *cs++;
  *rr++ = w18;
  *rr++ = w20;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<3; ++i) *rr++ = *cs++;
  /* #94: @26 = @25' */
  casadi_copy(w25, 11, w26);
  /* #95: @27 = 
  [[1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 2000, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 200, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 500, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1500, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.75]] */
  casadi_copy(casadi_c0, 121, w27);
  /* #96: @1 = mac(@26,@27,@1) */
  for (i=0, rr=w1; i<11; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w26+j, tt=w27+i*11; k<11; ++k) *rr += ss[k*1]**tt++;
  /* #97: @0 = mac(@1,@25,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w25+i*11; k<11; ++k) *rr += ss[k*1]**tt++;
  /* #98: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int broken_cost_ext_cost_e_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int broken_cost_ext_cost_e_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real broken_cost_ext_cost_e_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* broken_cost_ext_cost_e_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* broken_cost_ext_cost_e_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* broken_cost_ext_cost_e_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* broken_cost_ext_cost_e_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 196;
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 196*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
