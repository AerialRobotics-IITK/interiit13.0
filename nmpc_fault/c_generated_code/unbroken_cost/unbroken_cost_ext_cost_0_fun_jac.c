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
  #define CASADI_PREFIX(ID) unbroken_cost_ext_cost_0_fun_jac_ ## ID
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
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
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
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};

static const casadi_real casadi_c0[225] = {80., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 80., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 400., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 600., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 50., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 50., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.};
static const casadi_real casadi_c1[16] = {1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};

/* unbroken_cost_ext_cost_0_fun_jac:(i0[13],i1[4],i2[],i3[17])->(o0,o1[17]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cr, *cs;
  casadi_real w0, *w1=w+2, *w2=w+17, *w3=w+20, *w4=w+23, w5, *w6=w+28, w7, w8, w9, w10, *w11=w+36, w12, w13, w14, w15, w16, w17, w18, *w19=w+47, w20, w21, w22, w23, w24, *w25=w+56, w26, w27, w28, *w29=w+63, *w30=w+66, w31, w32, w33, *w34=w+72, *w35=w+76, *w36=w+80, *w37=w+95, *w38=w+110, *w39=w+335, *w40=w+350, *w41=w+354, *w42=w+370, *w43=w+374, *w44=w+390, *w45=w+615, *w46=w+618, w47, *w48=w+622, w49;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(1x15) */
  casadi_clear(w1, 15);
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
  /* #18: @8 = @11[0] */
  for (rr=(&w8), ss=w11+0; ss!=w11+1; ss+=1) *rr++ = *ss;
  /* #19: @9 = (@5*@8) */
  w9  = (w5*w8);
  /* #20: @10 = @4[1] */
  for (rr=(&w10), ss=w4+1; ss!=w4+2; ss+=1) *rr++ = *ss;
  /* #21: @12 = @11[1] */
  for (rr=(&w12), ss=w11+1; ss!=w11+2; ss+=1) *rr++ = *ss;
  /* #22: @13 = (@10*@12) */
  w13  = (w10*w12);
  /* #23: @9 = (@9-@13) */
  w9 -= w13;
  /* #24: @13 = @4[2] */
  for (rr=(&w13), ss=w4+2; ss!=w4+3; ss+=1) *rr++ = *ss;
  /* #25: @14 = @11[2] */
  for (rr=(&w14), ss=w11+2; ss!=w11+3; ss+=1) *rr++ = *ss;
  /* #26: @15 = (@13*@14) */
  w15  = (w13*w14);
  /* #27: @9 = (@9-@15) */
  w9 -= w15;
  /* #28: @15 = @4[3] */
  for (rr=(&w15), ss=w4+3; ss!=w4+4; ss+=1) *rr++ = *ss;
  /* #29: @16 = @11[3] */
  for (rr=(&w16), ss=w11+3; ss!=w11+4; ss+=1) *rr++ = *ss;
  /* #30: @17 = (@15*@16) */
  w17  = (w15*w16);
  /* #31: @9 = (@9-@17) */
  w9 -= w17;
  /* #32: @17 = (@5*@16) */
  w17  = (w5*w16);
  /* #33: @18 = (@10*@14) */
  w18  = (w10*w14);
  /* #34: @17 = (@17+@18) */
  w17 += w18;
  /* #35: @18 = (@13*@12) */
  w18  = (w13*w12);
  /* #36: @17 = (@17-@18) */
  w17 -= w18;
  /* #37: @18 = (@15*@8) */
  w18  = (w15*w8);
  /* #38: @17 = (@17+@18) */
  w17 += w18;
  /* #39: @4 = vertcat(@9, @0, @0, @17) */
  rr=w4;
  *rr++ = w9;
  *rr++ = w0;
  *rr++ = w0;
  *rr++ = w17;
  /* #40: @18 = ||@4||_F */
  w18 = sqrt(casadi_dot(4, w4, w4));
  /* #41: @19 = (@4/@18) */
  for (i=0, rr=w19, cr=w4; i<4; ++i) (*rr++)  = ((*cr++)/w18);
  /* #42: @20 = @19[0] */
  for (rr=(&w20), ss=w19+0; ss!=w19+1; ss+=1) *rr++ = *ss;
  /* #43: @21 = @19[1] */
  for (rr=(&w21), ss=w19+1; ss!=w19+2; ss+=1) *rr++ = *ss;
  /* #44: @21 = (-@21) */
  w21 = (- w21 );
  /* #45: @22 = @19[2] */
  for (rr=(&w22), ss=w19+2; ss!=w19+3; ss+=1) *rr++ = *ss;
  /* #46: @22 = (-@22) */
  w22 = (- w22 );
  /* #47: @23 = @19[3] */
  for (rr=(&w23), ss=w19+3; ss!=w19+4; ss+=1) *rr++ = *ss;
  /* #48: @24 = (-@23) */
  w24 = (- w23 );
  /* #49: @25 = vertcat(@20, @21, @22, @24) */
  rr=w25;
  *rr++ = w20;
  *rr++ = w21;
  *rr++ = w22;
  *rr++ = w24;
  /* #50: @20 = dot(@19, @19) */
  w20 = casadi_dot(4, w19, w19);
  /* #51: @25 = (@25/@20) */
  for (i=0, rr=w25; i<4; ++i) (*rr++) /= w20;
  /* #52: @21 = @25[1] */
  for (rr=(&w21), ss=w25+1; ss!=w25+2; ss+=1) *rr++ = *ss;
  /* #53: @22 = (@9*@21) */
  w22  = (w9*w21);
  /* #54: @24 = (@5*@12) */
  w24  = (w5*w12);
  /* #55: @26 = (@10*@8) */
  w26  = (w10*w8);
  /* #56: @24 = (@24+@26) */
  w24 += w26;
  /* #57: @26 = (@13*@16) */
  w26  = (w13*w16);
  /* #58: @24 = (@24+@26) */
  w24 += w26;
  /* #59: @26 = (@15*@14) */
  w26  = (w15*w14);
  /* #60: @24 = (@24-@26) */
  w24 -= w26;
  /* #61: @26 = @25[0] */
  for (rr=(&w26), ss=w25+0; ss!=w25+1; ss+=1) *rr++ = *ss;
  /* #62: @27 = (@24*@26) */
  w27  = (w24*w26);
  /* #63: @22 = (@22+@27) */
  w22 += w27;
  /* #64: @14 = (@5*@14) */
  w14  = (w5*w14);
  /* #65: @16 = (@10*@16) */
  w16  = (w10*w16);
  /* #66: @14 = (@14-@16) */
  w14 -= w16;
  /* #67: @8 = (@13*@8) */
  w8  = (w13*w8);
  /* #68: @14 = (@14+@8) */
  w14 += w8;
  /* #69: @12 = (@15*@12) */
  w12  = (w15*w12);
  /* #70: @14 = (@14+@12) */
  w14 += w12;
  /* #71: @12 = @25[3] */
  for (rr=(&w12), ss=w25+3; ss!=w25+4; ss+=1) *rr++ = *ss;
  /* #72: @8 = (@14*@12) */
  w8  = (w14*w12);
  /* #73: @22 = (@22+@8) */
  w22 += w8;
  /* #74: @8 = @25[2] */
  for (rr=(&w8), ss=w25+2; ss!=w25+3; ss+=1) *rr++ = *ss;
  /* #75: @16 = (@17*@8) */
  w16  = (w17*w8);
  /* #76: @22 = (@22-@16) */
  w22 -= w16;
  /* #77: @16 = sq(@22) */
  w16 = casadi_sq( w22 );
  /* #78: @27 = (@9*@8) */
  w27  = (w9*w8);
  /* #79: @28 = (@24*@12) */
  w28  = (w24*w12);
  /* #80: @27 = (@27-@28) */
  w27 -= w28;
  /* #81: @28 = (@14*@26) */
  w28  = (w14*w26);
  /* #82: @27 = (@27+@28) */
  w27 += w28;
  /* #83: @28 = (@17*@21) */
  w28  = (w17*w21);
  /* #84: @27 = (@27+@28) */
  w27 += w28;
  /* #85: @28 = sq(@27) */
  w28 = casadi_sq( w27 );
  /* #86: @16 = (@16+@28) */
  w16 += w28;
  /* #87: @3 = input[0][2] */
  casadi_copy(arg[0] ? arg[0]+7 : 0, 3, w3);
  /* #88: @29 = input[3][2] */
  casadi_copy(arg[3] ? arg[3]+7 : 0, 3, w29);
  /* #89: @3 = (@3-@29) */
  for (i=0, rr=w3, cs=w29; i<3; ++i) (*rr++) -= (*cs++);
  /* #90: @29 = input[0][3] */
  casadi_copy(arg[0] ? arg[0]+10 : 0, 3, w29);
  /* #91: @30 = input[3][3] */
  casadi_copy(arg[3] ? arg[3]+10 : 0, 3, w30);
  /* #92: @29 = (@29-@30) */
  for (i=0, rr=w29, cs=w30; i<3; ++i) (*rr++) -= (*cs++);
  /* #93: @28 = input[1][0] */
  w28 = arg[1] ? arg[1][0] : 0;
  /* #94: @31 = input[1][1] */
  w31 = arg[1] ? arg[1][1] : 0;
  /* #95: @32 = input[1][2] */
  w32 = arg[1] ? arg[1][2] : 0;
  /* #96: @33 = input[1][3] */
  w33 = arg[1] ? arg[1][3] : 0;
  /* #97: @34 = vertcat(@28, @31, @32, @33) */
  rr=w34;
  *rr++ = w28;
  *rr++ = w31;
  *rr++ = w32;
  *rr++ = w33;
  /* #98: @35 = input[3][4] */
  casadi_copy(arg[3] ? arg[3]+13 : 0, 4, w35);
  /* #99: @35 = (@34-@35) */
  for (i=0, rr=w35, cr=w34, cs=w35; i<4; ++i) (*rr++)  = ((*cr++)-(*cs++));
  /* #100: @36 = vertcat(@2, @16, @23, @3, @29, @35) */
  rr=w36;
  for (i=0, cs=w2; i<3; ++i) *rr++ = *cs++;
  *rr++ = w16;
  *rr++ = w23;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w29; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w35; i<4; ++i) *rr++ = *cs++;
  /* #101: @37 = @36' */
  casadi_copy(w36, 15, w37);
  /* #102: @38 = 
  [[80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] */
  casadi_copy(casadi_c0, 225, w38);
  /* #103: @39 = mac(@37,@38,@1) */
  casadi_copy(w1, 15, w39);
  for (i=0, rr=w39; i<15; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w37+j, tt=w38+i*15; k<15; ++k) *rr += ss[k*1]**tt++;
  /* #104: @16 = mac(@39,@36,@0) */
  casadi_copy((&w0), 1, (&w16));
  for (i=0, rr=(&w16); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w39+j, tt=w36+i*15; k<15; ++k) *rr += ss[k*1]**tt++;
  /* #105: @35 = zeros(1x4) */
  casadi_clear(w35, 4);
  /* #106: @40 = @34' */
  casadi_copy(w34, 4, w40);
  /* #107: @41 = 
  [[1, 0, 0, 0], 
   [0, 1, 0, 0], 
   [0, 0, 1, 0], 
   [0, 0, 0, 1]] */
  casadi_copy(casadi_c1, 16, w41);
  /* #108: @42 = mac(@40,@41,@35) */
  casadi_copy(w35, 4, w42);
  for (i=0, rr=w42; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w40+j, tt=w41+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #109: @0 = mac(@42,@34,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w42+j, tt=w34+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #110: @16 = (@16+@0) */
  w16 += w0;
  /* #111: output[0][0] = @16 */
  if (res[0]) res[0][0] = w16;
  /* #112: @42 = @42' */
  /* #113: @43 = @41' */
  for (i=0, rr=w43, cs=w41; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #114: @35 = mac(@40,@43,@35) */
  for (i=0, rr=w35; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w40+j, tt=w43+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #115: @35 = @35' */
  /* #116: @42 = (@42+@35) */
  for (i=0, rr=w42, cs=w35; i<4; ++i) (*rr++) += (*cs++);
  /* #117: @39 = @39' */
  /* #118: {@2, @16, @0, @3, @29, @35} = vertsplit(@39) */
  casadi_copy(w39, 3, w2);
  w16 = w39[3];
  w0 = w39[4];
  casadi_copy(w39+5, 3, w3);
  casadi_copy(w39+8, 3, w29);
  casadi_copy(w39+11, 4, w35);
  /* #119: @42 = (@42+@35) */
  for (i=0, rr=w42, cs=w35; i<4; ++i) (*rr++) += (*cs++);
  /* #120: @44 = @38' */
  for (i=0, rr=w44, cs=w38; i<15; ++i) for (j=0; j<15; ++j) rr[i+j*15] = *cs++;
  /* #121: @1 = mac(@37,@44,@1) */
  for (i=0, rr=w1; i<15; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w37+j, tt=w44+i*15; k<15; ++k) *rr += ss[k*1]**tt++;
  /* #122: @1 = @1' */
  /* #123: {@30, @23, @28, @45, @46, @35} = vertsplit(@1) */
  casadi_copy(w1, 3, w30);
  w23 = w1[3];
  w28 = w1[4];
  casadi_copy(w1+5, 3, w45);
  casadi_copy(w1+8, 3, w46);
  casadi_copy(w1+11, 4, w35);
  /* #124: @42 = (@42+@35) */
  for (i=0, rr=w42, cs=w35; i<4; ++i) (*rr++) += (*cs++);
  /* #125: {@31, @32, @33, @47} = vertsplit(@42) */
  w31 = w42[0];
  w32 = w42[1];
  w33 = w42[2];
  w47 = w42[3];
  /* #126: output[1][0] = @31 */
  if (res[1]) res[1][0] = w31;
  /* #127: output[1][1] = @32 */
  if (res[1]) res[1][1] = w32;
  /* #128: output[1][2] = @33 */
  if (res[1]) res[1][2] = w33;
  /* #129: output[1][3] = @47 */
  if (res[1]) res[1][3] = w47;
  /* #130: @2 = (@2+@30) */
  for (i=0, rr=w2, cs=w30; i<3; ++i) (*rr++) += (*cs++);
  /* #131: output[1][4] = @2 */
  if (res[1]) casadi_copy(w2, 3, res[1]+4);
  /* #132: @11 = (@11/@7) */
  for (i=0, rr=w11; i<4; ++i) (*rr++) /= w7;
  /* #133: @11 = (-@11) */
  for (i=0, rr=w11, cs=w11; i<4; ++i) *rr++ = (- *cs++ );
  /* #134: @42 = zeros(4x1) */
  casadi_clear(w42, 4);
  /* #135: @27 = (2.*@27) */
  w27 = (2.* w27 );
  /* #136: @47 = (@27*@16) */
  w47  = (w27*w16);
  /* #137: @33 = (@26*@47) */
  w33  = (w26*w47);
  /* #138: @22 = (2.*@22) */
  w22 = (2.* w22 );
  /* #139: @16 = (@22*@16) */
  w16  = (w22*w16);
  /* #140: @32 = (@12*@16) */
  w32  = (w12*w16);
  /* #141: @33 = (@33+@32) */
  w33 += w32;
  /* #142: @32 = (@15*@33) */
  w32  = (w15*w33);
  /* #143: @35 = @42; (@35[1] += @32) */
  casadi_copy(w42, 4, w35);
  for (rr=w35+1, ss=(&w32); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #144: @32 = (@13*@33) */
  w32  = (w13*w33);
  /* #145: (@35[0] += @32) */
  for (rr=w35+0, ss=(&w32); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #146: @32 = (@10*@33) */
  w32  = (w10*w33);
  /* #147: @32 = (-@32) */
  w32 = (- w32 );
  /* #148: (@35[3] += @32) */
  for (rr=w35+3, ss=(&w32); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #149: @33 = (@5*@33) */
  w33  = (w5*w33);
  /* #150: (@35[2] += @33) */
  for (rr=w35+2, ss=(&w33); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #151: @33 = (@26*@16) */
  w33  = (w26*w16);
  /* #152: @32 = (@12*@47) */
  w32  = (w12*w47);
  /* #153: @33 = (@33-@32) */
  w33 -= w32;
  /* #154: @32 = (@15*@33) */
  w32  = (w15*w33);
  /* #155: @32 = (-@32) */
  w32 = (- w32 );
  /* #156: (@35[2] += @32) */
  for (rr=w35+2, ss=(&w32); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #157: @32 = (@13*@33) */
  w32  = (w13*w33);
  /* #158: (@35[3] += @32) */
  for (rr=w35+3, ss=(&w32); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #159: @32 = (@10*@33) */
  w32  = (w10*w33);
  /* #160: (@35[0] += @32) */
  for (rr=w35+0, ss=(&w32); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #161: @33 = (@5*@33) */
  w33  = (w5*w33);
  /* #162: (@35[1] += @33) */
  for (rr=w35+1, ss=(&w33); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #163: @33 = (@21*@47) */
  w33  = (w21*w47);
  /* #164: @32 = (@8*@16) */
  w32  = (w8*w16);
  /* #165: @33 = (@33-@32) */
  w33 -= w32;
  /* #166: @40 = @42; (@40[3] += @0) */
  casadi_copy(w42, 4, w40);
  for (rr=w40+3, ss=(&w0); rr!=w40+4; rr+=1) *rr += *ss++;
  /* #167: @25 = (@25/@20) */
  for (i=0, rr=w25; i<4; ++i) (*rr++) /= w20;
  /* #168: @25 = (-@25) */
  for (i=0, rr=w25, cs=w25; i<4; ++i) *rr++ = (- *cs++ );
  /* #169: @0 = (@17*@47) */
  w0  = (w17*w47);
  /* #170: @34 = @42; (@34[1] += @0) */
  casadi_copy(w42, 4, w34);
  for (rr=w34+1, ss=(&w0); rr!=w34+2; rr+=1) *rr += *ss++;
  /* #171: @0 = (@14*@47) */
  w0  = (w14*w47);
  /* #172: (@34[0] += @0) */
  for (rr=w34+0, ss=(&w0); rr!=w34+1; rr+=1) *rr += *ss++;
  /* #173: @0 = (@24*@47) */
  w0  = (w24*w47);
  /* #174: @0 = (-@0) */
  w0 = (- w0 );
  /* #175: (@34[3] += @0) */
  for (rr=w34+3, ss=(&w0); rr!=w34+4; rr+=1) *rr += *ss++;
  /* #176: @0 = (@9*@47) */
  w0  = (w9*w47);
  /* #177: (@34[2] += @0) */
  for (rr=w34+2, ss=(&w0); rr!=w34+3; rr+=1) *rr += *ss++;
  /* #178: @0 = (@17*@16) */
  w0  = (w17*w16);
  /* #179: @0 = (-@0) */
  w0 = (- w0 );
  /* #180: (@34[2] += @0) */
  for (rr=w34+2, ss=(&w0); rr!=w34+3; rr+=1) *rr += *ss++;
  /* #181: @0 = (@14*@16) */
  w0  = (w14*w16);
  /* #182: (@34[3] += @0) */
  for (rr=w34+3, ss=(&w0); rr!=w34+4; rr+=1) *rr += *ss++;
  /* #183: @0 = (@24*@16) */
  w0  = (w24*w16);
  /* #184: (@34[0] += @0) */
  for (rr=w34+0, ss=(&w0); rr!=w34+1; rr+=1) *rr += *ss++;
  /* #185: @0 = (@9*@16) */
  w0  = (w9*w16);
  /* #186: (@34[1] += @0) */
  for (rr=w34+1, ss=(&w0); rr!=w34+2; rr+=1) *rr += *ss++;
  /* #187: @0 = dot(@25, @34) */
  w0 = casadi_dot(4, w25, w34);
  /* #188: @48 = (@0*@19) */
  for (i=0, rr=w48, cs=w19; i<4; ++i) (*rr++)  = (w0*(*cs++));
  /* #189: @40 = (@40+@48) */
  for (i=0, rr=w40, cs=w48; i<4; ++i) (*rr++) += (*cs++);
  /* #190: @40 = (@40+@48) */
  for (i=0, rr=w40, cs=w48; i<4; ++i) (*rr++) += (*cs++);
  /* #191: @34 = (@34/@20) */
  for (i=0, rr=w34; i<4; ++i) (*rr++) /= w20;
  /* #192: {@0, @32, @31, @49} = vertsplit(@34) */
  w0 = w34[0];
  w32 = w34[1];
  w31 = w34[2];
  w49 = w34[3];
  /* #193: @49 = (-@49) */
  w49 = (- w49 );
  /* #194: (@40[3] += @49) */
  for (rr=w40+3, ss=(&w49); rr!=w40+4; rr+=1) *rr += *ss++;
  /* #195: @31 = (-@31) */
  w31 = (- w31 );
  /* #196: (@40[2] += @31) */
  for (rr=w40+2, ss=(&w31); rr!=w40+3; rr+=1) *rr += *ss++;
  /* #197: @32 = (-@32) */
  w32 = (- w32 );
  /* #198: (@40[1] += @32) */
  for (rr=w40+1, ss=(&w32); rr!=w40+2; rr+=1) *rr += *ss++;
  /* #199: (@40[0] += @0) */
  for (rr=w40+0, ss=(&w0); rr!=w40+1; rr+=1) *rr += *ss++;
  /* #200: @34 = (@40/@18) */
  for (i=0, rr=w34, cr=w40; i<4; ++i) (*rr++)  = ((*cr++)/w18);
  /* #201: @48 = (@19/@18) */
  for (i=0, rr=w48, cr=w19; i<4; ++i) (*rr++)  = ((*cr++)/w18);
  /* #202: @48 = (-@48) */
  for (i=0, rr=w48, cs=w48; i<4; ++i) *rr++ = (- *cs++ );
  /* #203: @0 = dot(@48, @40) */
  w0 = casadi_dot(4, w48, w40);
  /* #204: @0 = (@0/@18) */
  w0 /= w18;
  /* #205: @40 = (@0*@4) */
  for (i=0, rr=w40, cs=w4; i<4; ++i) (*rr++)  = (w0*(*cs++));
  /* #206: @34 = (@34+@40) */
  for (i=0, rr=w34, cs=w40; i<4; ++i) (*rr++) += (*cs++);
  /* #207: {@0, NULL, NULL, @32} = vertsplit(@34) */
  w0 = w34[0];
  w32 = w34[3];
  /* #208: @33 = (@33+@32) */
  w33 += w32;
  /* #209: @32 = (@15*@33) */
  w32  = (w15*w33);
  /* #210: (@35[0] += @32) */
  for (rr=w35+0, ss=(&w32); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #211: @32 = (@13*@33) */
  w32  = (w13*w33);
  /* #212: @32 = (-@32) */
  w32 = (- w32 );
  /* #213: (@35[1] += @32) */
  for (rr=w35+1, ss=(&w32); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #214: @32 = (@10*@33) */
  w32  = (w10*w33);
  /* #215: (@35[2] += @32) */
  for (rr=w35+2, ss=(&w32); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #216: @33 = (@5*@33) */
  w33  = (w5*w33);
  /* #217: (@35[3] += @33) */
  for (rr=w35+3, ss=(&w33); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #218: @47 = (@8*@47) */
  w47  = (w8*w47);
  /* #219: @16 = (@21*@16) */
  w16  = (w21*w16);
  /* #220: @47 = (@47+@16) */
  w47 += w16;
  /* #221: @47 = (@47+@0) */
  w47 += w0;
  /* #222: @0 = (@15*@47) */
  w0  = (w15*w47);
  /* #223: @0 = (-@0) */
  w0 = (- w0 );
  /* #224: (@35[3] += @0) */
  for (rr=w35+3, ss=(&w0); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #225: @0 = (@13*@47) */
  w0  = (w13*w47);
  /* #226: @0 = (-@0) */
  w0 = (- w0 );
  /* #227: (@35[2] += @0) */
  for (rr=w35+2, ss=(&w0); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #228: @0 = (@10*@47) */
  w0  = (w10*w47);
  /* #229: @0 = (-@0) */
  w0 = (- w0 );
  /* #230: (@35[1] += @0) */
  for (rr=w35+1, ss=(&w0); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #231: @47 = (@5*@47) */
  w47  = (w5*w47);
  /* #232: (@35[0] += @47) */
  for (rr=w35+0, ss=(&w47); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #233: @47 = dot(@11, @35) */
  w47 = casadi_dot(4, w11, w35);
  /* #234: @34 = (@47*@6) */
  for (i=0, rr=w34, cs=w6; i<4; ++i) (*rr++)  = (w47*(*cs++));
  /* #235: @34 = (2.*@34) */
  for (i=0, rr=w34, cs=w34; i<4; ++i) *rr++ = (2.* *cs++ );
  /* #236: @35 = (@35/@7) */
  for (i=0, rr=w35; i<4; ++i) (*rr++) /= w7;
  /* #237: {@47, @0, @16, @33} = vertsplit(@35) */
  w47 = w35[0];
  w0 = w35[1];
  w16 = w35[2];
  w33 = w35[3];
  /* #238: @33 = (-@33) */
  w33 = (- w33 );
  /* #239: (@34[3] += @33) */
  for (rr=w34+3, ss=(&w33); rr!=w34+4; rr+=1) *rr += *ss++;
  /* #240: @16 = (-@16) */
  w16 = (- w16 );
  /* #241: (@34[2] += @16) */
  for (rr=w34+2, ss=(&w16); rr!=w34+3; rr+=1) *rr += *ss++;
  /* #242: @0 = (-@0) */
  w0 = (- w0 );
  /* #243: (@34[1] += @0) */
  for (rr=w34+1, ss=(&w0); rr!=w34+2; rr+=1) *rr += *ss++;
  /* #244: (@34[0] += @47) */
  for (rr=w34+0, ss=(&w47); rr!=w34+1; rr+=1) *rr += *ss++;
  /* #245: @27 = (@27*@23) */
  w27 *= w23;
  /* #246: @47 = (@26*@27) */
  w47  = (w26*w27);
  /* #247: @22 = (@22*@23) */
  w22 *= w23;
  /* #248: @23 = (@12*@22) */
  w23  = (w12*w22);
  /* #249: @47 = (@47+@23) */
  w47 += w23;
  /* #250: @23 = (@15*@47) */
  w23  = (w15*w47);
  /* #251: @35 = @42; (@35[1] += @23) */
  casadi_copy(w42, 4, w35);
  for (rr=w35+1, ss=(&w23); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #252: @23 = (@13*@47) */
  w23  = (w13*w47);
  /* #253: (@35[0] += @23) */
  for (rr=w35+0, ss=(&w23); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #254: @23 = (@10*@47) */
  w23  = (w10*w47);
  /* #255: @23 = (-@23) */
  w23 = (- w23 );
  /* #256: (@35[3] += @23) */
  for (rr=w35+3, ss=(&w23); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #257: @47 = (@5*@47) */
  w47  = (w5*w47);
  /* #258: (@35[2] += @47) */
  for (rr=w35+2, ss=(&w47); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #259: @26 = (@26*@22) */
  w26 *= w22;
  /* #260: @12 = (@12*@27) */
  w12 *= w27;
  /* #261: @26 = (@26-@12) */
  w26 -= w12;
  /* #262: @12 = (@15*@26) */
  w12  = (w15*w26);
  /* #263: @12 = (-@12) */
  w12 = (- w12 );
  /* #264: (@35[2] += @12) */
  for (rr=w35+2, ss=(&w12); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #265: @12 = (@13*@26) */
  w12  = (w13*w26);
  /* #266: (@35[3] += @12) */
  for (rr=w35+3, ss=(&w12); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #267: @12 = (@10*@26) */
  w12  = (w10*w26);
  /* #268: (@35[0] += @12) */
  for (rr=w35+0, ss=(&w12); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #269: @26 = (@5*@26) */
  w26  = (w5*w26);
  /* #270: (@35[1] += @26) */
  for (rr=w35+1, ss=(&w26); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #271: @26 = (@21*@27) */
  w26  = (w21*w27);
  /* #272: @12 = (@8*@22) */
  w12  = (w8*w22);
  /* #273: @26 = (@26-@12) */
  w26 -= w12;
  /* #274: @40 = @42; (@40[3] += @28) */
  casadi_copy(w42, 4, w40);
  for (rr=w40+3, ss=(&w28); rr!=w40+4; rr+=1) *rr += *ss++;
  /* #275: @28 = (@17*@27) */
  w28  = (w17*w27);
  /* #276: (@42[1] += @28) */
  for (rr=w42+1, ss=(&w28); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #277: @28 = (@14*@27) */
  w28  = (w14*w27);
  /* #278: (@42[0] += @28) */
  for (rr=w42+0, ss=(&w28); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #279: @28 = (@24*@27) */
  w28  = (w24*w27);
  /* #280: @28 = (-@28) */
  w28 = (- w28 );
  /* #281: (@42[3] += @28) */
  for (rr=w42+3, ss=(&w28); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #282: @28 = (@9*@27) */
  w28  = (w9*w27);
  /* #283: (@42[2] += @28) */
  for (rr=w42+2, ss=(&w28); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #284: @17 = (@17*@22) */
  w17 *= w22;
  /* #285: @17 = (-@17) */
  w17 = (- w17 );
  /* #286: (@42[2] += @17) */
  for (rr=w42+2, ss=(&w17); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #287: @14 = (@14*@22) */
  w14 *= w22;
  /* #288: (@42[3] += @14) */
  for (rr=w42+3, ss=(&w14); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #289: @24 = (@24*@22) */
  w24 *= w22;
  /* #290: (@42[0] += @24) */
  for (rr=w42+0, ss=(&w24); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #291: @9 = (@9*@22) */
  w9 *= w22;
  /* #292: (@42[1] += @9) */
  for (rr=w42+1, ss=(&w9); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #293: @9 = dot(@25, @42) */
  w9 = casadi_dot(4, w25, w42);
  /* #294: @19 = (@9*@19) */
  for (i=0, rr=w19, cs=w19; i<4; ++i) (*rr++)  = (w9*(*cs++));
  /* #295: @40 = (@40+@19) */
  for (i=0, rr=w40, cs=w19; i<4; ++i) (*rr++) += (*cs++);
  /* #296: @40 = (@40+@19) */
  for (i=0, rr=w40, cs=w19; i<4; ++i) (*rr++) += (*cs++);
  /* #297: @42 = (@42/@20) */
  for (i=0, rr=w42; i<4; ++i) (*rr++) /= w20;
  /* #298: {@20, @9, @24, @14} = vertsplit(@42) */
  w20 = w42[0];
  w9 = w42[1];
  w24 = w42[2];
  w14 = w42[3];
  /* #299: @14 = (-@14) */
  w14 = (- w14 );
  /* #300: (@40[3] += @14) */
  for (rr=w40+3, ss=(&w14); rr!=w40+4; rr+=1) *rr += *ss++;
  /* #301: @24 = (-@24) */
  w24 = (- w24 );
  /* #302: (@40[2] += @24) */
  for (rr=w40+2, ss=(&w24); rr!=w40+3; rr+=1) *rr += *ss++;
  /* #303: @9 = (-@9) */
  w9 = (- w9 );
  /* #304: (@40[1] += @9) */
  for (rr=w40+1, ss=(&w9); rr!=w40+2; rr+=1) *rr += *ss++;
  /* #305: (@40[0] += @20) */
  for (rr=w40+0, ss=(&w20); rr!=w40+1; rr+=1) *rr += *ss++;
  /* #306: @42 = (@40/@18) */
  for (i=0, rr=w42, cr=w40; i<4; ++i) (*rr++)  = ((*cr++)/w18);
  /* #307: @20 = dot(@48, @40) */
  w20 = casadi_dot(4, w48, w40);
  /* #308: @20 = (@20/@18) */
  w20 /= w18;
  /* #309: @4 = (@20*@4) */
  for (i=0, rr=w4, cs=w4; i<4; ++i) (*rr++)  = (w20*(*cs++));
  /* #310: @42 = (@42+@4) */
  for (i=0, rr=w42, cs=w4; i<4; ++i) (*rr++) += (*cs++);
  /* #311: {@20, NULL, NULL, @18} = vertsplit(@42) */
  w20 = w42[0];
  w18 = w42[3];
  /* #312: @26 = (@26+@18) */
  w26 += w18;
  /* #313: @18 = (@15*@26) */
  w18  = (w15*w26);
  /* #314: (@35[0] += @18) */
  for (rr=w35+0, ss=(&w18); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #315: @18 = (@13*@26) */
  w18  = (w13*w26);
  /* #316: @18 = (-@18) */
  w18 = (- w18 );
  /* #317: (@35[1] += @18) */
  for (rr=w35+1, ss=(&w18); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #318: @18 = (@10*@26) */
  w18  = (w10*w26);
  /* #319: (@35[2] += @18) */
  for (rr=w35+2, ss=(&w18); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #320: @26 = (@5*@26) */
  w26  = (w5*w26);
  /* #321: (@35[3] += @26) */
  for (rr=w35+3, ss=(&w26); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #322: @8 = (@8*@27) */
  w8 *= w27;
  /* #323: @21 = (@21*@22) */
  w21 *= w22;
  /* #324: @8 = (@8+@21) */
  w8 += w21;
  /* #325: @8 = (@8+@20) */
  w8 += w20;
  /* #326: @15 = (@15*@8) */
  w15 *= w8;
  /* #327: @15 = (-@15) */
  w15 = (- w15 );
  /* #328: (@35[3] += @15) */
  for (rr=w35+3, ss=(&w15); rr!=w35+4; rr+=1) *rr += *ss++;
  /* #329: @13 = (@13*@8) */
  w13 *= w8;
  /* #330: @13 = (-@13) */
  w13 = (- w13 );
  /* #331: (@35[2] += @13) */
  for (rr=w35+2, ss=(&w13); rr!=w35+3; rr+=1) *rr += *ss++;
  /* #332: @10 = (@10*@8) */
  w10 *= w8;
  /* #333: @10 = (-@10) */
  w10 = (- w10 );
  /* #334: (@35[1] += @10) */
  for (rr=w35+1, ss=(&w10); rr!=w35+2; rr+=1) *rr += *ss++;
  /* #335: @5 = (@5*@8) */
  w5 *= w8;
  /* #336: (@35[0] += @5) */
  for (rr=w35+0, ss=(&w5); rr!=w35+1; rr+=1) *rr += *ss++;
  /* #337: @5 = dot(@11, @35) */
  w5 = casadi_dot(4, w11, w35);
  /* #338: @6 = (@5*@6) */
  for (i=0, rr=w6, cs=w6; i<4; ++i) (*rr++)  = (w5*(*cs++));
  /* #339: @34 = (@34+@6) */
  for (i=0, rr=w34, cs=w6; i<4; ++i) (*rr++) += (*cs++);
  /* #340: @34 = (@34+@6) */
  for (i=0, rr=w34, cs=w6; i<4; ++i) (*rr++) += (*cs++);
  /* #341: @35 = (@35/@7) */
  for (i=0, rr=w35; i<4; ++i) (*rr++) /= w7;
  /* #342: {@7, @5, @8, @10} = vertsplit(@35) */
  w7 = w35[0];
  w5 = w35[1];
  w8 = w35[2];
  w10 = w35[3];
  /* #343: @10 = (-@10) */
  w10 = (- w10 );
  /* #344: (@34[3] += @10) */
  for (rr=w34+3, ss=(&w10); rr!=w34+4; rr+=1) *rr += *ss++;
  /* #345: @8 = (-@8) */
  w8 = (- w8 );
  /* #346: (@34[2] += @8) */
  for (rr=w34+2, ss=(&w8); rr!=w34+3; rr+=1) *rr += *ss++;
  /* #347: @5 = (-@5) */
  w5 = (- w5 );
  /* #348: (@34[1] += @5) */
  for (rr=w34+1, ss=(&w5); rr!=w34+2; rr+=1) *rr += *ss++;
  /* #349: (@34[0] += @7) */
  for (rr=w34+0, ss=(&w7); rr!=w34+1; rr+=1) *rr += *ss++;
  /* #350: output[1][5] = @34 */
  if (res[1]) casadi_copy(w34, 4, res[1]+7);
  /* #351: @3 = (@3+@45) */
  for (i=0, rr=w3, cs=w45; i<3; ++i) (*rr++) += (*cs++);
  /* #352: output[1][6] = @3 */
  if (res[1]) casadi_copy(w3, 3, res[1]+11);
  /* #353: @29 = (@29+@46) */
  for (i=0, rr=w29, cs=w46; i<3; ++i) (*rr++) += (*cs++);
  /* #354: output[1][7] = @29 */
  if (res[1]) casadi_copy(w29, 3, res[1]+14);
  return 0;
}

CASADI_SYMBOL_EXPORT int unbroken_cost_ext_cost_0_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int unbroken_cost_ext_cost_0_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int unbroken_cost_ext_cost_0_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void unbroken_cost_ext_cost_0_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int unbroken_cost_ext_cost_0_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void unbroken_cost_ext_cost_0_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void unbroken_cost_ext_cost_0_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void unbroken_cost_ext_cost_0_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int unbroken_cost_ext_cost_0_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int unbroken_cost_ext_cost_0_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real unbroken_cost_ext_cost_0_fun_jac_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* unbroken_cost_ext_cost_0_fun_jac_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* unbroken_cost_ext_cost_0_fun_jac_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* unbroken_cost_ext_cost_0_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* unbroken_cost_ext_cost_0_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int unbroken_cost_ext_cost_0_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 10;
  if (sz_res) *sz_res = 8;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 627;
  return 0;
}

CASADI_SYMBOL_EXPORT int unbroken_cost_ext_cost_0_fun_jac_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 10*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 8*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 627*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
