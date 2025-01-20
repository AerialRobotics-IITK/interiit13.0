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
  #define CASADI_PREFIX(ID) broken_expl_ode_fun_ ## ID
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
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_c3 CASADI_PREFIX(c3)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
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

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const casadi_real casadi_c0[3] = {0., 0., 9.8100000000000005e+00};
static const casadi_real casadi_c1[4] = {-1.7399999999999999e-01, 1.7399999999999999e-01, 1.7399999999999999e-01, -1.7399999999999999e-01};
static const casadi_real casadi_c2[4] = {1.7399999999999999e-01, -1.7399999999999999e-01, 1.7399999999999999e-01, -1.7399999999999999e-01};
static const casadi_real casadi_c3[4] = {1.6000000000000000e-02, 1.6000000000000000e-02, -1.6000000000000000e-02, -1.6000000000000000e-02};

/* broken_expl_ode_fun:(i0[13],i1[4],i2[17])->(o0[13]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+4, w1, *w2=w+8, w3, w4, w5, w6, w7, w8, w9, *w10=w+19, *w11=w+23, *w12=w+27, *w13=w+31, *w14=w+35, *w15=w+51, *w16=w+67, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, *w27=w+80, *w28=w+83, *w29=w+86, *w30=w+89, *w31=w+98;
  /* #0: @0 = input[0][2] */
  casadi_copy(arg[0] ? arg[0]+7 : 0, 3, w0);
  /* #1: output[0][0] = @0 */
  casadi_copy(w0, 3, res[0]);
  /* #2: @1 = 0.5 */
  w1 = 5.0000000000000000e-01;
  /* #3: @2 = zeros(4x1) */
  casadi_clear(w2, 4);
  /* #4: @3 = 0 */
  w3 = 0.;
  /* #5: @0 = input[0][3] */
  casadi_copy(arg[0] ? arg[0]+10 : 0, 3, w0);
  /* #6: @4 = @0[0] */
  for (rr=(&w4), ss=w0+0; ss!=w0+1; ss+=1) *rr++ = *ss;
  /* #7: @5 = (-@4) */
  w5 = (- w4 );
  /* #8: @6 = @0[1] */
  for (rr=(&w6), ss=w0+1; ss!=w0+2; ss+=1) *rr++ = *ss;
  /* #9: @7 = (-@6) */
  w7 = (- w6 );
  /* #10: @8 = @0[2] */
  for (rr=(&w8), ss=w0+2; ss!=w0+3; ss+=1) *rr++ = *ss;
  /* #11: @9 = (-@8) */
  w9 = (- w8 );
  /* #12: @10 = horzcat(@3, @5, @7, @9) */
  rr=w10;
  *rr++ = w3;
  *rr++ = w5;
  *rr++ = w7;
  *rr++ = w9;
  /* #13: @10 = @10' */
  /* #14: @11 = horzcat(@4, @3, @8, @7) */
  rr=w11;
  *rr++ = w4;
  *rr++ = w3;
  *rr++ = w8;
  *rr++ = w7;
  /* #15: @11 = @11' */
  /* #16: @12 = horzcat(@6, @9, @3, @4) */
  rr=w12;
  *rr++ = w6;
  *rr++ = w9;
  *rr++ = w3;
  *rr++ = w4;
  /* #17: @12 = @12' */
  /* #18: @13 = horzcat(@8, @6, @5, @3) */
  rr=w13;
  *rr++ = w8;
  *rr++ = w6;
  *rr++ = w5;
  *rr++ = w3;
  /* #19: @13 = @13' */
  /* #20: @14 = horzcat(@10, @11, @12, @13) */
  rr=w14;
  for (i=0, cs=w10; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w11; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w12; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w13; i<4; ++i) *rr++ = *cs++;
  /* #21: @15 = @14' */
  for (i=0, rr=w15, cs=w14; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #22: @10 = input[0][1] */
  casadi_copy(arg[0] ? arg[0]+3 : 0, 4, w10);
  /* #23: @2 = mac(@15,@10,@2) */
  for (i=0, rr=w2; i<1; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w15+j, tt=w10+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #24: @2 = (@1*@2) */
  for (i=0, rr=w2, cs=w2; i<4; ++i) (*rr++)  = (w1*(*cs++));
  /* #25: output[0][1] = @2 */
  if (res[0]) casadi_copy(w2, 4, res[0]+3);
  /* #26: @0 = [0, 0, 9.81] */
  casadi_copy(casadi_c0, 3, w0);
  /* #27: @16 = zeros(3x1) */
  casadi_clear(w16, 3);
  /* #28: @1 = 1 */
  w1 = 1.;
  /* #29: @5 = @10[2] */
  for (rr=(&w5), ss=w10+2; ss!=w10+3; ss+=1) *rr++ = *ss;
  /* #30: @9 = sq(@5) */
  w9 = casadi_sq( w5 );
  /* #31: @7 = @10[3] */
  for (rr=(&w7), ss=w10+3; ss!=w10+4; ss+=1) *rr++ = *ss;
  /* #32: @17 = sq(@7) */
  w17 = casadi_sq( w7 );
  /* #33: @18 = (@9+@17) */
  w18  = (w9+w17);
  /* #34: @18 = (2.*@18) */
  w18 = (2.* w18 );
  /* #35: @18 = (@1-@18) */
  w18  = (w1-w18);
  /* #36: @19 = @10[1] */
  for (rr=(&w19), ss=w10+1; ss!=w10+2; ss+=1) *rr++ = *ss;
  /* #37: @20 = (@19*@5) */
  w20  = (w19*w5);
  /* #38: @21 = @10[0] */
  for (rr=(&w21), ss=w10+0; ss!=w10+1; ss+=1) *rr++ = *ss;
  /* #39: @22 = (@21*@7) */
  w22  = (w21*w7);
  /* #40: @23 = (@20-@22) */
  w23  = (w20-w22);
  /* #41: @23 = (2.*@23) */
  w23 = (2.* w23 );
  /* #42: @24 = (@19*@7) */
  w24  = (w19*w7);
  /* #43: @25 = (@21*@5) */
  w25  = (w21*w5);
  /* #44: @26 = (@24+@25) */
  w26  = (w24+w25);
  /* #45: @26 = (2.*@26) */
  w26 = (2.* w26 );
  /* #46: @27 = horzcat(@18, @23, @26) */
  rr=w27;
  *rr++ = w18;
  *rr++ = w23;
  *rr++ = w26;
  /* #47: @27 = @27' */
  /* #48: @20 = (@20+@22) */
  w20 += w22;
  /* #49: @20 = (2.*@20) */
  w20 = (2.* w20 );
  /* #50: @22 = sq(@19) */
  w22 = casadi_sq( w19 );
  /* #51: @17 = (@22+@17) */
  w17  = (w22+w17);
  /* #52: @17 = (2.*@17) */
  w17 = (2.* w17 );
  /* #53: @17 = (@1-@17) */
  w17  = (w1-w17);
  /* #54: @5 = (@5*@7) */
  w5 *= w7;
  /* #55: @21 = (@21*@19) */
  w21 *= w19;
  /* #56: @19 = (@5-@21) */
  w19  = (w5-w21);
  /* #57: @19 = (2.*@19) */
  w19 = (2.* w19 );
  /* #58: @28 = horzcat(@20, @17, @19) */
  rr=w28;
  *rr++ = w20;
  *rr++ = w17;
  *rr++ = w19;
  /* #59: @28 = @28' */
  /* #60: @24 = (@24-@25) */
  w24 -= w25;
  /* #61: @24 = (2.*@24) */
  w24 = (2.* w24 );
  /* #62: @5 = (@5+@21) */
  w5 += w21;
  /* #63: @5 = (2.*@5) */
  w5 = (2.* w5 );
  /* #64: @22 = (@22+@9) */
  w22 += w9;
  /* #65: @22 = (2.*@22) */
  w22 = (2.* w22 );
  /* #66: @1 = (@1-@22) */
  w1 -= w22;
  /* #67: @29 = horzcat(@24, @5, @1) */
  rr=w29;
  *rr++ = w24;
  *rr++ = w5;
  *rr++ = w1;
  /* #68: @29 = @29' */
  /* #69: @30 = horzcat(@27, @28, @29) */
  rr=w30;
  for (i=0, cs=w27; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w28; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w29; i<3; ++i) *rr++ = *cs++;
  /* #70: @31 = @30' */
  for (i=0, rr=w31, cs=w30; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #71: @24 = 17 */
  w24 = 17.;
  /* #72: @5 = input[1][0] */
  w5 = arg[1] ? arg[1][0] : 0;
  /* #73: @1 = input[1][1] */
  w1 = arg[1] ? arg[1][1] : 0;
  /* #74: @22 = input[1][2] */
  w22 = arg[1] ? arg[1][2] : 0;
  /* #75: @9 = input[1][3] */
  w9 = arg[1] ? arg[1][3] : 0;
  /* #76: @10 = vertcat(@5, @1, @22, @9) */
  rr=w10;
  *rr++ = w5;
  *rr++ = w1;
  *rr++ = w22;
  *rr++ = w9;
  /* #77: @10 = (@24*@10) */
  for (i=0, rr=w10, cs=w10; i<4; ++i) (*rr++)  = (w24*(*cs++));
  /* #78: @24 = @10[0] */
  for (rr=(&w24), ss=w10+0; ss!=w10+1; ss+=1) *rr++ = *ss;
  /* #79: @24 = (-@24) */
  w24 = (- w24 );
  /* #80: @5 = @10[1] */
  for (rr=(&w5), ss=w10+1; ss!=w10+2; ss+=1) *rr++ = *ss;
  /* #81: @24 = (@24-@5) */
  w24 -= w5;
  /* #82: @5 = @10[2] */
  for (rr=(&w5), ss=w10+2; ss!=w10+3; ss+=1) *rr++ = *ss;
  /* #83: @24 = (@24-@5) */
  w24 -= w5;
  /* #84: @5 = @10[3] */
  for (rr=(&w5), ss=w10+3; ss!=w10+4; ss+=1) *rr++ = *ss;
  /* #85: @24 = (@24-@5) */
  w24 -= w5;
  /* #86: @27 = vertcat(@3, @3, @24) */
  rr=w27;
  *rr++ = w3;
  *rr++ = w3;
  *rr++ = w24;
  /* #87: @24 = 2.06431 */
  w24 = 2.0643076919999999e+00;
  /* #88: @27 = (@27/@24) */
  for (i=0, rr=w27; i<3; ++i) (*rr++) /= w24;
  /* #89: @16 = mac(@31,@27,@16) */
  for (i=0, rr=w16; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w31+j, tt=w27+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #90: @0 = (@0+@16) */
  for (i=0, rr=w0, cs=w16; i<3; ++i) (*rr++) += (*cs++);
  /* #91: output[0][2] = @0 */
  if (res[0]) casadi_copy(w0, 3, res[0]+7);
  /* #92: @10 = @10' */
  /* #93: @2 = [-0.174, 0.174, 0.174, -0.174] */
  casadi_copy(casadi_c1, 4, w2);
  /* #94: @24 = mac(@10,@2,@3) */
  casadi_copy((&w3), 1, (&w24));
  for (i=0, rr=(&w24); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w10+j, tt=w2+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #95: @5 = -0.0358798 */
  w5 = -3.5879797087419341e-02;
  /* #96: @5 = (@5*@6) */
  w5 *= w6;
  /* #97: @5 = (@5*@8) */
  w5 *= w8;
  /* #98: @24 = (@24+@5) */
  w24 += w5;
  /* #99: @5 = 0.028 */
  w5 = 2.8000000000000001e-02;
  /* #100: @24 = (@24/@5) */
  w24 /= w5;
  /* #101: output[0][3] = @24 */
  if (res[0]) res[0][10] = w24;
  /* #102: @2 = [0.174, -0.174, 0.174, -0.174] */
  casadi_copy(casadi_c2, 4, w2);
  /* #103: @24 = mac(@10,@2,@3) */
  casadi_copy((&w3), 1, (&w24));
  for (i=0, rr=(&w24); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w10+j, tt=w2+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #104: @6 = 0.0358798 */
  w6 = 3.5879797087419341e-02;
  /* #105: @6 = (@6*@8) */
  w6 *= w8;
  /* #106: @6 = (@6*@4) */
  w6 *= w4;
  /* #107: @24 = (@24+@6) */
  w24 += w6;
  /* #108: @24 = (@24/@5) */
  w24 /= w5;
  /* #109: output[0][4] = @24 */
  if (res[0]) res[0][11] = w24;
  /* #110: @2 = [0.016, 0.016, -0.016, -0.016] */
  casadi_copy(casadi_c3, 4, w2);
  /* #111: @3 = mac(@10,@2,@3) */
  for (i=0, rr=(&w3); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w10+j, tt=w2+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #112: @24 = 0.0638798 */
  w24 = 6.3879797087419338e-02;
  /* #113: @3 = (@3/@24) */
  w3 /= w24;
  /* #114: output[0][5] = @3 */
  if (res[0]) res[0][12] = w3;
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int broken_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void broken_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int broken_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void broken_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void broken_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void broken_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int broken_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int broken_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real broken_expl_ode_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* broken_expl_ode_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* broken_expl_ode_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* broken_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* broken_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int broken_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 107;
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_expl_ode_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 107*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
