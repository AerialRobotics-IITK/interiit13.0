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
  #define CASADI_PREFIX(ID) broken_cost_ext_cost_e_fun_jac_ ## ID
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

/* broken_cost_ext_cost_e_fun_jac:(i0[13],i1[],i2[],i3[17])->(o0,o1[13]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cr, *cs;
  casadi_real w0, *w1=w+2, *w2=w+13, *w3=w+16, *w4=w+19, w5, *w6=w+24, w7, w8, w9, w10, *w11=w+32, w12, w13, w14, w15, w16, w17, w18, *w19=w+43, w20, w21, w22, w23, w24, *w25=w+52, w26, w27, w28, *w29=w+59, *w30=w+62, *w31=w+65, *w32=w+76, *w33=w+87, *w34=w+208, *w35=w+219, *w36=w+340, *w37=w+343, *w38=w+346, w39, w40, w41, *w42=w+353, *w43=w+357, *w44=w+361, *w45=w+365, w46, w47;
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
  /* #93: @31 = vertcat(@2, @16, @23, @3, @29) */
  rr=w31;
  for (i=0, cs=w2; i<3; ++i) *rr++ = *cs++;
  *rr++ = w16;
  *rr++ = w23;
  for (i=0, cs=w3; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w29; i<3; ++i) *rr++ = *cs++;
  /* #94: @32 = @31' */
  casadi_copy(w31, 11, w32);
  /* #95: @33 = 
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
  casadi_copy(casadi_c0, 121, w33);
  /* #96: @34 = mac(@32,@33,@1) */
  casadi_copy(w1, 11, w34);
  for (i=0, rr=w34; i<11; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w32+j, tt=w33+i*11; k<11; ++k) *rr += ss[k*1]**tt++;
  /* #97: @0 = mac(@34,@31,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w34+j, tt=w31+i*11; k<11; ++k) *rr += ss[k*1]**tt++;
  /* #98: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #99: @34 = @34' */
  /* #100: {@2, @0, @16, @3, @29} = vertsplit(@34) */
  casadi_copy(w34, 3, w2);
  w0 = w34[3];
  w16 = w34[4];
  casadi_copy(w34+5, 3, w3);
  casadi_copy(w34+8, 3, w29);
  /* #101: @35 = @33' */
  for (i=0, rr=w35, cs=w33; i<11; ++i) for (j=0; j<11; ++j) rr[i+j*11] = *cs++;
  /* #102: @1 = mac(@32,@35,@1) */
  for (i=0, rr=w1; i<11; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w32+j, tt=w35+i*11; k<11; ++k) *rr += ss[k*1]**tt++;
  /* #103: @1 = @1' */
  /* #104: {@30, @23, @28, @36, @37} = vertsplit(@1) */
  casadi_copy(w1, 3, w30);
  w23 = w1[3];
  w28 = w1[4];
  casadi_copy(w1+5, 3, w36);
  casadi_copy(w1+8, 3, w37);
  /* #105: @2 = (@2+@30) */
  for (i=0, rr=w2, cs=w30; i<3; ++i) (*rr++) += (*cs++);
  /* #106: output[1][0] = @2 */
  casadi_copy(w2, 3, res[1]);
  /* #107: @11 = (@11/@7) */
  for (i=0, rr=w11; i<4; ++i) (*rr++) /= w7;
  /* #108: @11 = (-@11) */
  for (i=0, rr=w11, cs=w11; i<4; ++i) *rr++ = (- *cs++ );
  /* #109: @38 = zeros(4x1) */
  casadi_clear(w38, 4);
  /* #110: @27 = (2.*@27) */
  w27 = (2.* w27 );
  /* #111: @39 = (@27*@0) */
  w39  = (w27*w0);
  /* #112: @40 = (@26*@39) */
  w40  = (w26*w39);
  /* #113: @22 = (2.*@22) */
  w22 = (2.* w22 );
  /* #114: @0 = (@22*@0) */
  w0  = (w22*w0);
  /* #115: @41 = (@12*@0) */
  w41  = (w12*w0);
  /* #116: @40 = (@40+@41) */
  w40 += w41;
  /* #117: @41 = (@15*@40) */
  w41  = (w15*w40);
  /* #118: @42 = @38; (@42[1] += @41) */
  casadi_copy(w38, 4, w42);
  for (rr=w42+1, ss=(&w41); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #119: @41 = (@13*@40) */
  w41  = (w13*w40);
  /* #120: (@42[0] += @41) */
  for (rr=w42+0, ss=(&w41); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #121: @41 = (@10*@40) */
  w41  = (w10*w40);
  /* #122: @41 = (-@41) */
  w41 = (- w41 );
  /* #123: (@42[3] += @41) */
  for (rr=w42+3, ss=(&w41); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #124: @40 = (@5*@40) */
  w40  = (w5*w40);
  /* #125: (@42[2] += @40) */
  for (rr=w42+2, ss=(&w40); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #126: @40 = (@26*@0) */
  w40  = (w26*w0);
  /* #127: @41 = (@12*@39) */
  w41  = (w12*w39);
  /* #128: @40 = (@40-@41) */
  w40 -= w41;
  /* #129: @41 = (@15*@40) */
  w41  = (w15*w40);
  /* #130: @41 = (-@41) */
  w41 = (- w41 );
  /* #131: (@42[2] += @41) */
  for (rr=w42+2, ss=(&w41); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #132: @41 = (@13*@40) */
  w41  = (w13*w40);
  /* #133: (@42[3] += @41) */
  for (rr=w42+3, ss=(&w41); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #134: @41 = (@10*@40) */
  w41  = (w10*w40);
  /* #135: (@42[0] += @41) */
  for (rr=w42+0, ss=(&w41); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #136: @40 = (@5*@40) */
  w40  = (w5*w40);
  /* #137: (@42[1] += @40) */
  for (rr=w42+1, ss=(&w40); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #138: @40 = (@21*@39) */
  w40  = (w21*w39);
  /* #139: @41 = (@8*@0) */
  w41  = (w8*w0);
  /* #140: @40 = (@40-@41) */
  w40 -= w41;
  /* #141: @43 = @38; (@43[3] += @16) */
  casadi_copy(w38, 4, w43);
  for (rr=w43+3, ss=(&w16); rr!=w43+4; rr+=1) *rr += *ss++;
  /* #142: @25 = (@25/@20) */
  for (i=0, rr=w25; i<4; ++i) (*rr++) /= w20;
  /* #143: @25 = (-@25) */
  for (i=0, rr=w25, cs=w25; i<4; ++i) *rr++ = (- *cs++ );
  /* #144: @16 = (@17*@39) */
  w16  = (w17*w39);
  /* #145: @44 = @38; (@44[1] += @16) */
  casadi_copy(w38, 4, w44);
  for (rr=w44+1, ss=(&w16); rr!=w44+2; rr+=1) *rr += *ss++;
  /* #146: @16 = (@14*@39) */
  w16  = (w14*w39);
  /* #147: (@44[0] += @16) */
  for (rr=w44+0, ss=(&w16); rr!=w44+1; rr+=1) *rr += *ss++;
  /* #148: @16 = (@24*@39) */
  w16  = (w24*w39);
  /* #149: @16 = (-@16) */
  w16 = (- w16 );
  /* #150: (@44[3] += @16) */
  for (rr=w44+3, ss=(&w16); rr!=w44+4; rr+=1) *rr += *ss++;
  /* #151: @16 = (@9*@39) */
  w16  = (w9*w39);
  /* #152: (@44[2] += @16) */
  for (rr=w44+2, ss=(&w16); rr!=w44+3; rr+=1) *rr += *ss++;
  /* #153: @16 = (@17*@0) */
  w16  = (w17*w0);
  /* #154: @16 = (-@16) */
  w16 = (- w16 );
  /* #155: (@44[2] += @16) */
  for (rr=w44+2, ss=(&w16); rr!=w44+3; rr+=1) *rr += *ss++;
  /* #156: @16 = (@14*@0) */
  w16  = (w14*w0);
  /* #157: (@44[3] += @16) */
  for (rr=w44+3, ss=(&w16); rr!=w44+4; rr+=1) *rr += *ss++;
  /* #158: @16 = (@24*@0) */
  w16  = (w24*w0);
  /* #159: (@44[0] += @16) */
  for (rr=w44+0, ss=(&w16); rr!=w44+1; rr+=1) *rr += *ss++;
  /* #160: @16 = (@9*@0) */
  w16  = (w9*w0);
  /* #161: (@44[1] += @16) */
  for (rr=w44+1, ss=(&w16); rr!=w44+2; rr+=1) *rr += *ss++;
  /* #162: @16 = dot(@25, @44) */
  w16 = casadi_dot(4, w25, w44);
  /* #163: @45 = (@16*@19) */
  for (i=0, rr=w45, cs=w19; i<4; ++i) (*rr++)  = (w16*(*cs++));
  /* #164: @43 = (@43+@45) */
  for (i=0, rr=w43, cs=w45; i<4; ++i) (*rr++) += (*cs++);
  /* #165: @43 = (@43+@45) */
  for (i=0, rr=w43, cs=w45; i<4; ++i) (*rr++) += (*cs++);
  /* #166: @44 = (@44/@20) */
  for (i=0, rr=w44; i<4; ++i) (*rr++) /= w20;
  /* #167: {@16, @41, @46, @47} = vertsplit(@44) */
  w16 = w44[0];
  w41 = w44[1];
  w46 = w44[2];
  w47 = w44[3];
  /* #168: @47 = (-@47) */
  w47 = (- w47 );
  /* #169: (@43[3] += @47) */
  for (rr=w43+3, ss=(&w47); rr!=w43+4; rr+=1) *rr += *ss++;
  /* #170: @46 = (-@46) */
  w46 = (- w46 );
  /* #171: (@43[2] += @46) */
  for (rr=w43+2, ss=(&w46); rr!=w43+3; rr+=1) *rr += *ss++;
  /* #172: @41 = (-@41) */
  w41 = (- w41 );
  /* #173: (@43[1] += @41) */
  for (rr=w43+1, ss=(&w41); rr!=w43+2; rr+=1) *rr += *ss++;
  /* #174: (@43[0] += @16) */
  for (rr=w43+0, ss=(&w16); rr!=w43+1; rr+=1) *rr += *ss++;
  /* #175: @44 = (@43/@18) */
  for (i=0, rr=w44, cr=w43; i<4; ++i) (*rr++)  = ((*cr++)/w18);
  /* #176: @45 = (@19/@18) */
  for (i=0, rr=w45, cr=w19; i<4; ++i) (*rr++)  = ((*cr++)/w18);
  /* #177: @45 = (-@45) */
  for (i=0, rr=w45, cs=w45; i<4; ++i) *rr++ = (- *cs++ );
  /* #178: @16 = dot(@45, @43) */
  w16 = casadi_dot(4, w45, w43);
  /* #179: @16 = (@16/@18) */
  w16 /= w18;
  /* #180: @43 = (@16*@4) */
  for (i=0, rr=w43, cs=w4; i<4; ++i) (*rr++)  = (w16*(*cs++));
  /* #181: @44 = (@44+@43) */
  for (i=0, rr=w44, cs=w43; i<4; ++i) (*rr++) += (*cs++);
  /* #182: {@16, NULL, NULL, @41} = vertsplit(@44) */
  w16 = w44[0];
  w41 = w44[3];
  /* #183: @40 = (@40+@41) */
  w40 += w41;
  /* #184: @41 = (@15*@40) */
  w41  = (w15*w40);
  /* #185: (@42[0] += @41) */
  for (rr=w42+0, ss=(&w41); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #186: @41 = (@13*@40) */
  w41  = (w13*w40);
  /* #187: @41 = (-@41) */
  w41 = (- w41 );
  /* #188: (@42[1] += @41) */
  for (rr=w42+1, ss=(&w41); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #189: @41 = (@10*@40) */
  w41  = (w10*w40);
  /* #190: (@42[2] += @41) */
  for (rr=w42+2, ss=(&w41); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #191: @40 = (@5*@40) */
  w40  = (w5*w40);
  /* #192: (@42[3] += @40) */
  for (rr=w42+3, ss=(&w40); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #193: @39 = (@8*@39) */
  w39  = (w8*w39);
  /* #194: @0 = (@21*@0) */
  w0  = (w21*w0);
  /* #195: @39 = (@39+@0) */
  w39 += w0;
  /* #196: @39 = (@39+@16) */
  w39 += w16;
  /* #197: @16 = (@15*@39) */
  w16  = (w15*w39);
  /* #198: @16 = (-@16) */
  w16 = (- w16 );
  /* #199: (@42[3] += @16) */
  for (rr=w42+3, ss=(&w16); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #200: @16 = (@13*@39) */
  w16  = (w13*w39);
  /* #201: @16 = (-@16) */
  w16 = (- w16 );
  /* #202: (@42[2] += @16) */
  for (rr=w42+2, ss=(&w16); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #203: @16 = (@10*@39) */
  w16  = (w10*w39);
  /* #204: @16 = (-@16) */
  w16 = (- w16 );
  /* #205: (@42[1] += @16) */
  for (rr=w42+1, ss=(&w16); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #206: @39 = (@5*@39) */
  w39  = (w5*w39);
  /* #207: (@42[0] += @39) */
  for (rr=w42+0, ss=(&w39); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #208: @39 = dot(@11, @42) */
  w39 = casadi_dot(4, w11, w42);
  /* #209: @44 = (@39*@6) */
  for (i=0, rr=w44, cs=w6; i<4; ++i) (*rr++)  = (w39*(*cs++));
  /* #210: @44 = (2.*@44) */
  for (i=0, rr=w44, cs=w44; i<4; ++i) *rr++ = (2.* *cs++ );
  /* #211: @42 = (@42/@7) */
  for (i=0, rr=w42; i<4; ++i) (*rr++) /= w7;
  /* #212: {@39, @16, @0, @40} = vertsplit(@42) */
  w39 = w42[0];
  w16 = w42[1];
  w0 = w42[2];
  w40 = w42[3];
  /* #213: @40 = (-@40) */
  w40 = (- w40 );
  /* #214: (@44[3] += @40) */
  for (rr=w44+3, ss=(&w40); rr!=w44+4; rr+=1) *rr += *ss++;
  /* #215: @0 = (-@0) */
  w0 = (- w0 );
  /* #216: (@44[2] += @0) */
  for (rr=w44+2, ss=(&w0); rr!=w44+3; rr+=1) *rr += *ss++;
  /* #217: @16 = (-@16) */
  w16 = (- w16 );
  /* #218: (@44[1] += @16) */
  for (rr=w44+1, ss=(&w16); rr!=w44+2; rr+=1) *rr += *ss++;
  /* #219: (@44[0] += @39) */
  for (rr=w44+0, ss=(&w39); rr!=w44+1; rr+=1) *rr += *ss++;
  /* #220: @27 = (@27*@23) */
  w27 *= w23;
  /* #221: @39 = (@26*@27) */
  w39  = (w26*w27);
  /* #222: @22 = (@22*@23) */
  w22 *= w23;
  /* #223: @23 = (@12*@22) */
  w23  = (w12*w22);
  /* #224: @39 = (@39+@23) */
  w39 += w23;
  /* #225: @23 = (@15*@39) */
  w23  = (w15*w39);
  /* #226: @42 = @38; (@42[1] += @23) */
  casadi_copy(w38, 4, w42);
  for (rr=w42+1, ss=(&w23); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #227: @23 = (@13*@39) */
  w23  = (w13*w39);
  /* #228: (@42[0] += @23) */
  for (rr=w42+0, ss=(&w23); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #229: @23 = (@10*@39) */
  w23  = (w10*w39);
  /* #230: @23 = (-@23) */
  w23 = (- w23 );
  /* #231: (@42[3] += @23) */
  for (rr=w42+3, ss=(&w23); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #232: @39 = (@5*@39) */
  w39  = (w5*w39);
  /* #233: (@42[2] += @39) */
  for (rr=w42+2, ss=(&w39); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #234: @26 = (@26*@22) */
  w26 *= w22;
  /* #235: @12 = (@12*@27) */
  w12 *= w27;
  /* #236: @26 = (@26-@12) */
  w26 -= w12;
  /* #237: @12 = (@15*@26) */
  w12  = (w15*w26);
  /* #238: @12 = (-@12) */
  w12 = (- w12 );
  /* #239: (@42[2] += @12) */
  for (rr=w42+2, ss=(&w12); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #240: @12 = (@13*@26) */
  w12  = (w13*w26);
  /* #241: (@42[3] += @12) */
  for (rr=w42+3, ss=(&w12); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #242: @12 = (@10*@26) */
  w12  = (w10*w26);
  /* #243: (@42[0] += @12) */
  for (rr=w42+0, ss=(&w12); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #244: @26 = (@5*@26) */
  w26  = (w5*w26);
  /* #245: (@42[1] += @26) */
  for (rr=w42+1, ss=(&w26); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #246: @26 = (@21*@27) */
  w26  = (w21*w27);
  /* #247: @12 = (@8*@22) */
  w12  = (w8*w22);
  /* #248: @26 = (@26-@12) */
  w26 -= w12;
  /* #249: @43 = @38; (@43[3] += @28) */
  casadi_copy(w38, 4, w43);
  for (rr=w43+3, ss=(&w28); rr!=w43+4; rr+=1) *rr += *ss++;
  /* #250: @28 = (@17*@27) */
  w28  = (w17*w27);
  /* #251: (@38[1] += @28) */
  for (rr=w38+1, ss=(&w28); rr!=w38+2; rr+=1) *rr += *ss++;
  /* #252: @28 = (@14*@27) */
  w28  = (w14*w27);
  /* #253: (@38[0] += @28) */
  for (rr=w38+0, ss=(&w28); rr!=w38+1; rr+=1) *rr += *ss++;
  /* #254: @28 = (@24*@27) */
  w28  = (w24*w27);
  /* #255: @28 = (-@28) */
  w28 = (- w28 );
  /* #256: (@38[3] += @28) */
  for (rr=w38+3, ss=(&w28); rr!=w38+4; rr+=1) *rr += *ss++;
  /* #257: @28 = (@9*@27) */
  w28  = (w9*w27);
  /* #258: (@38[2] += @28) */
  for (rr=w38+2, ss=(&w28); rr!=w38+3; rr+=1) *rr += *ss++;
  /* #259: @17 = (@17*@22) */
  w17 *= w22;
  /* #260: @17 = (-@17) */
  w17 = (- w17 );
  /* #261: (@38[2] += @17) */
  for (rr=w38+2, ss=(&w17); rr!=w38+3; rr+=1) *rr += *ss++;
  /* #262: @14 = (@14*@22) */
  w14 *= w22;
  /* #263: (@38[3] += @14) */
  for (rr=w38+3, ss=(&w14); rr!=w38+4; rr+=1) *rr += *ss++;
  /* #264: @24 = (@24*@22) */
  w24 *= w22;
  /* #265: (@38[0] += @24) */
  for (rr=w38+0, ss=(&w24); rr!=w38+1; rr+=1) *rr += *ss++;
  /* #266: @9 = (@9*@22) */
  w9 *= w22;
  /* #267: (@38[1] += @9) */
  for (rr=w38+1, ss=(&w9); rr!=w38+2; rr+=1) *rr += *ss++;
  /* #268: @9 = dot(@25, @38) */
  w9 = casadi_dot(4, w25, w38);
  /* #269: @19 = (@9*@19) */
  for (i=0, rr=w19, cs=w19; i<4; ++i) (*rr++)  = (w9*(*cs++));
  /* #270: @43 = (@43+@19) */
  for (i=0, rr=w43, cs=w19; i<4; ++i) (*rr++) += (*cs++);
  /* #271: @43 = (@43+@19) */
  for (i=0, rr=w43, cs=w19; i<4; ++i) (*rr++) += (*cs++);
  /* #272: @38 = (@38/@20) */
  for (i=0, rr=w38; i<4; ++i) (*rr++) /= w20;
  /* #273: {@20, @9, @24, @14} = vertsplit(@38) */
  w20 = w38[0];
  w9 = w38[1];
  w24 = w38[2];
  w14 = w38[3];
  /* #274: @14 = (-@14) */
  w14 = (- w14 );
  /* #275: (@43[3] += @14) */
  for (rr=w43+3, ss=(&w14); rr!=w43+4; rr+=1) *rr += *ss++;
  /* #276: @24 = (-@24) */
  w24 = (- w24 );
  /* #277: (@43[2] += @24) */
  for (rr=w43+2, ss=(&w24); rr!=w43+3; rr+=1) *rr += *ss++;
  /* #278: @9 = (-@9) */
  w9 = (- w9 );
  /* #279: (@43[1] += @9) */
  for (rr=w43+1, ss=(&w9); rr!=w43+2; rr+=1) *rr += *ss++;
  /* #280: (@43[0] += @20) */
  for (rr=w43+0, ss=(&w20); rr!=w43+1; rr+=1) *rr += *ss++;
  /* #281: @38 = (@43/@18) */
  for (i=0, rr=w38, cr=w43; i<4; ++i) (*rr++)  = ((*cr++)/w18);
  /* #282: @20 = dot(@45, @43) */
  w20 = casadi_dot(4, w45, w43);
  /* #283: @20 = (@20/@18) */
  w20 /= w18;
  /* #284: @4 = (@20*@4) */
  for (i=0, rr=w4, cs=w4; i<4; ++i) (*rr++)  = (w20*(*cs++));
  /* #285: @38 = (@38+@4) */
  for (i=0, rr=w38, cs=w4; i<4; ++i) (*rr++) += (*cs++);
  /* #286: {@20, NULL, NULL, @18} = vertsplit(@38) */
  w20 = w38[0];
  w18 = w38[3];
  /* #287: @26 = (@26+@18) */
  w26 += w18;
  /* #288: @18 = (@15*@26) */
  w18  = (w15*w26);
  /* #289: (@42[0] += @18) */
  for (rr=w42+0, ss=(&w18); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #290: @18 = (@13*@26) */
  w18  = (w13*w26);
  /* #291: @18 = (-@18) */
  w18 = (- w18 );
  /* #292: (@42[1] += @18) */
  for (rr=w42+1, ss=(&w18); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #293: @18 = (@10*@26) */
  w18  = (w10*w26);
  /* #294: (@42[2] += @18) */
  for (rr=w42+2, ss=(&w18); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #295: @26 = (@5*@26) */
  w26  = (w5*w26);
  /* #296: (@42[3] += @26) */
  for (rr=w42+3, ss=(&w26); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #297: @8 = (@8*@27) */
  w8 *= w27;
  /* #298: @21 = (@21*@22) */
  w21 *= w22;
  /* #299: @8 = (@8+@21) */
  w8 += w21;
  /* #300: @8 = (@8+@20) */
  w8 += w20;
  /* #301: @15 = (@15*@8) */
  w15 *= w8;
  /* #302: @15 = (-@15) */
  w15 = (- w15 );
  /* #303: (@42[3] += @15) */
  for (rr=w42+3, ss=(&w15); rr!=w42+4; rr+=1) *rr += *ss++;
  /* #304: @13 = (@13*@8) */
  w13 *= w8;
  /* #305: @13 = (-@13) */
  w13 = (- w13 );
  /* #306: (@42[2] += @13) */
  for (rr=w42+2, ss=(&w13); rr!=w42+3; rr+=1) *rr += *ss++;
  /* #307: @10 = (@10*@8) */
  w10 *= w8;
  /* #308: @10 = (-@10) */
  w10 = (- w10 );
  /* #309: (@42[1] += @10) */
  for (rr=w42+1, ss=(&w10); rr!=w42+2; rr+=1) *rr += *ss++;
  /* #310: @5 = (@5*@8) */
  w5 *= w8;
  /* #311: (@42[0] += @5) */
  for (rr=w42+0, ss=(&w5); rr!=w42+1; rr+=1) *rr += *ss++;
  /* #312: @5 = dot(@11, @42) */
  w5 = casadi_dot(4, w11, w42);
  /* #313: @6 = (@5*@6) */
  for (i=0, rr=w6, cs=w6; i<4; ++i) (*rr++)  = (w5*(*cs++));
  /* #314: @44 = (@44+@6) */
  for (i=0, rr=w44, cs=w6; i<4; ++i) (*rr++) += (*cs++);
  /* #315: @44 = (@44+@6) */
  for (i=0, rr=w44, cs=w6; i<4; ++i) (*rr++) += (*cs++);
  /* #316: @42 = (@42/@7) */
  for (i=0, rr=w42; i<4; ++i) (*rr++) /= w7;
  /* #317: {@7, @5, @8, @10} = vertsplit(@42) */
  w7 = w42[0];
  w5 = w42[1];
  w8 = w42[2];
  w10 = w42[3];
  /* #318: @10 = (-@10) */
  w10 = (- w10 );
  /* #319: (@44[3] += @10) */
  for (rr=w44+3, ss=(&w10); rr!=w44+4; rr+=1) *rr += *ss++;
  /* #320: @8 = (-@8) */
  w8 = (- w8 );
  /* #321: (@44[2] += @8) */
  for (rr=w44+2, ss=(&w8); rr!=w44+3; rr+=1) *rr += *ss++;
  /* #322: @5 = (-@5) */
  w5 = (- w5 );
  /* #323: (@44[1] += @5) */
  for (rr=w44+1, ss=(&w5); rr!=w44+2; rr+=1) *rr += *ss++;
  /* #324: (@44[0] += @7) */
  for (rr=w44+0, ss=(&w7); rr!=w44+1; rr+=1) *rr += *ss++;
  /* #325: output[1][1] = @44 */
  if (res[1]) casadi_copy(w44, 4, res[1]+3);
  /* #326: @3 = (@3+@36) */
  for (i=0, rr=w3, cs=w36; i<3; ++i) (*rr++) += (*cs++);
  /* #327: output[1][2] = @3 */
  if (res[1]) casadi_copy(w3, 3, res[1]+7);
  /* #328: @29 = (@29+@37) */
  for (i=0, rr=w29, cs=w37; i<3; ++i) (*rr++) += (*cs++);
  /* #329: output[1][3] = @29 */
  if (res[1]) casadi_copy(w29, 3, res[1]+10);
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void broken_cost_ext_cost_e_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int broken_cost_ext_cost_e_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int broken_cost_ext_cost_e_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real broken_cost_ext_cost_e_fun_jac_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* broken_cost_ext_cost_e_fun_jac_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* broken_cost_ext_cost_e_fun_jac_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* broken_cost_ext_cost_e_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* broken_cost_ext_cost_e_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9;
  if (sz_res) *sz_res = 7;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 371;
  return 0;
}

CASADI_SYMBOL_EXPORT int broken_cost_ext_cost_e_fun_jac_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 7*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 371*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
