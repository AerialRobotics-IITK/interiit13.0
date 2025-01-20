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
  #define CASADI_PREFIX(ID) quad_expl_vde_adj_ ## ID
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
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
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

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s3[18] = {17, 1, 0, 14, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const casadi_real casadi_c0[4] = {-1.6000000000000000e-02, 1.6000000000000000e-02, -1.6000000000000000e-02, 1.6000000000000000e-02};
static const casadi_real casadi_c1[4] = {1.7394826817189071e-01, -1.7394826817189071e-01, -1.7394826817189071e-01, 1.7394826817189071e-01};
static const casadi_real casadi_c2[4] = {-1.7394826817189071e-01, -1.7394826817189071e-01, 1.7394826817189071e-01, 1.7394826817189071e-01};

/* quad_expl_vde_adj:(i0[13],i1[13],i2[4],i3[17])->(o0[17x1,14nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real *w0=w+4, *w1=w+8, w2, *w3=w+13, *w4=w+22, *w5=w+35, *w6=w+38, *w7=w+42, w8, w9, w10, w11, w12, w13, w14, w15, w16, *w17=w+54, *w18=w+58, *w19=w+61, *w20=w+70, *w21=w+73, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31, w32, *w33=w+87, *w34=w+91, *w35=w+95, *w36=w+99, *w37=w+103, *w38=w+119;
  /* #0: @0 = zeros(4x1) */
  casadi_clear(w0, 4);
  /* #1: @1 = input[0][1] */
  casadi_copy(arg[0] ? arg[0]+3 : 0, 4, w1);
  /* #2: @2 = @1[1] */
  for (rr=(&w2), ss=w1+1; ss!=w1+2; ss+=1) *rr++ = *ss;
  /* #3: @3 = zeros(3x3) */
  casadi_clear(w3, 9);
  /* #4: @4 = input[1][0] */
  casadi_copy(arg[1], 13, w4);
  /* #5: {@5, @6, @7, @8, @9, @10} = vertsplit(@4) */
  casadi_copy(w4, 3, w5);
  casadi_copy(w4+3, 4, w6);
  casadi_copy(w4+7, 3, w7);
  w8 = w4[10];
  w9 = w4[11];
  w10 = w4[12];
  /* #6: @11 = 0 */
  w11 = 0.;
  /* #7: @12 = 8.54858 */
  w12 = 8.5485799999999994e+00;
  /* #8: @13 = input[2][0] */
  w13 = arg[2] ? arg[2][0] : 0;
  /* #9: @14 = input[2][1] */
  w14 = arg[2] ? arg[2][1] : 0;
  /* #10: @15 = input[2][2] */
  w15 = arg[2] ? arg[2][2] : 0;
  /* #11: @16 = input[2][3] */
  w16 = arg[2] ? arg[2][3] : 0;
  /* #12: @17 = vertcat(@13, @14, @15, @16) */
  rr=w17;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w16;
  /* #13: @17 = (@12*@17) */
  for (i=0, rr=w17, cs=w17; i<4; ++i) (*rr++)  = (w12*(*cs++));
  /* #14: @13 = @17[0] */
  for (rr=(&w13), ss=w17+0; ss!=w17+1; ss+=1) *rr++ = *ss;
  /* #15: @14 = @17[1] */
  for (rr=(&w14), ss=w17+1; ss!=w17+2; ss+=1) *rr++ = *ss;
  /* #16: @13 = (@13+@14) */
  w13 += w14;
  /* #17: @14 = @17[2] */
  for (rr=(&w14), ss=w17+2; ss!=w17+3; ss+=1) *rr++ = *ss;
  /* #18: @13 = (@13+@14) */
  w13 += w14;
  /* #19: @14 = @17[3] */
  for (rr=(&w14), ss=w17+3; ss!=w17+4; ss+=1) *rr++ = *ss;
  /* #20: @13 = (@13+@14) */
  w13 += w14;
  /* #21: @18 = vertcat(@11, @11, @13) */
  rr=w18;
  *rr++ = w11;
  *rr++ = w11;
  *rr++ = w13;
  /* #22: @13 = 2.064 */
  w13 = 2.0640000000000001e+00;
  /* #23: @18 = (@18/@13) */
  for (i=0, rr=w18; i<3; ++i) (*rr++) /= w13;
  /* #24: @18 = @18' */
  /* #25: @3 = mac(@7,@18,@3) */
  for (i=0, rr=w3; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w7+j, tt=w18+i*1; k<1; ++k) *rr += ss[k*3]**tt++;
  /* #26: @19 = @3' */
  for (i=0, rr=w19, cs=w3; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #27: {@18, @20, @21} = horzsplit(@19) */
  casadi_copy(w19, 3, w18);
  casadi_copy(w19+3, 3, w20);
  casadi_copy(w19+6, 3, w21);
  /* #28: @21 = @21' */
  /* #29: {@13, @14, @15} = horzsplit(@21) */
  w13 = w21[0];
  w14 = w21[1];
  w15 = w21[2];
  /* #30: @14 = (2.*@14) */
  w14 = (2.* w14 );
  /* #31: @16 = (@2*@14) */
  w16  = (w2*w14);
  /* #32: @22 = @1[2] */
  for (rr=(&w22), ss=w1+2; ss!=w1+3; ss+=1) *rr++ = *ss;
  /* #33: @13 = (2.*@13) */
  w13 = (2.* w13 );
  /* #34: @23 = (@22*@13) */
  w23  = (w22*w13);
  /* #35: @16 = (@16-@23) */
  w16 -= w23;
  /* #36: @20 = @20' */
  /* #37: {@23, @24, @25} = horzsplit(@20) */
  w23 = w20[0];
  w24 = w20[1];
  w25 = w20[2];
  /* #38: @25 = (2.*@25) */
  w25 = (2.* w25 );
  /* #39: @26 = (@2*@25) */
  w26  = (w2*w25);
  /* #40: @16 = (@16-@26) */
  w16 -= w26;
  /* #41: @26 = @1[3] */
  for (rr=(&w26), ss=w1+3; ss!=w1+4; ss+=1) *rr++ = *ss;
  /* #42: @23 = (2.*@23) */
  w23 = (2.* w23 );
  /* #43: @27 = (@26*@23) */
  w27  = (w26*w23);
  /* #44: @16 = (@16+@27) */
  w16 += w27;
  /* #45: @18 = @18' */
  /* #46: {@27, @28, @29} = horzsplit(@18) */
  w27 = w18[0];
  w28 = w18[1];
  w29 = w18[2];
  /* #47: @29 = (2.*@29) */
  w29 = (2.* w29 );
  /* #48: @30 = (@22*@29) */
  w30  = (w22*w29);
  /* #49: @16 = (@16+@30) */
  w16 += w30;
  /* #50: @28 = (2.*@28) */
  w28 = (2.* w28 );
  /* #51: @30 = (@26*@28) */
  w30  = (w26*w28);
  /* #52: @16 = (@16-@30) */
  w16 -= w30;
  /* #53: @17 = @0; (@17[0] += @16) */
  casadi_copy(w0, 4, w17);
  for (rr=w17+0, ss=(&w16); rr!=w17+1; rr+=1) *rr += *ss++;
  /* #54: @16 = (2.*@2) */
  w16 = (2.* w2 );
  /* #55: @15 = (-@15) */
  w15 = (- w15 );
  /* #56: @15 = (2.*@15) */
  w15 = (2.* w15 );
  /* #57: @30 = (@16*@15) */
  w30  = (w16*w15);
  /* #58: @31 = @1[0] */
  for (rr=(&w31), ss=w1+0; ss!=w1+1; ss+=1) *rr++ = *ss;
  /* #59: @32 = (@31*@14) */
  w32  = (w31*w14);
  /* #60: @30 = (@30+@32) */
  w30 += w32;
  /* #61: @32 = (@26*@13) */
  w32  = (w26*w13);
  /* #62: @30 = (@30+@32) */
  w30 += w32;
  /* #63: @32 = (@31*@25) */
  w32  = (w31*w25);
  /* #64: @30 = (@30-@32) */
  w30 -= w32;
  /* #65: @24 = (-@24) */
  w24 = (- w24 );
  /* #66: @24 = (2.*@24) */
  w24 = (2.* w24 );
  /* #67: @16 = (@16*@24) */
  w16 *= w24;
  /* #68: @30 = (@30+@16) */
  w30 += w16;
  /* #69: @16 = (@22*@23) */
  w16  = (w22*w23);
  /* #70: @30 = (@30+@16) */
  w30 += w16;
  /* #71: @16 = (@26*@29) */
  w16  = (w26*w29);
  /* #72: @30 = (@30+@16) */
  w30 += w16;
  /* #73: @16 = (@22*@28) */
  w16  = (w22*w28);
  /* #74: @30 = (@30+@16) */
  w30 += w16;
  /* #75: (@17[1] += @30) */
  for (rr=w17+1, ss=(&w30); rr!=w17+2; rr+=1) *rr += *ss++;
  /* #76: @30 = (@22*@14) */
  w30  = (w22*w14);
  /* #77: @16 = (@2*@13) */
  w16  = (w2*w13);
  /* #78: @30 = (@30+@16) */
  w30 += w16;
  /* #79: @16 = (@22*@25) */
  w16  = (w22*w25);
  /* #80: @30 = (@30+@16) */
  w30 += w16;
  /* #81: @16 = (2.*@26) */
  w16 = (2.* w26 );
  /* #82: @24 = (@16*@24) */
  w24  = (w16*w24);
  /* #83: @30 = (@30+@24) */
  w30 += w24;
  /* #84: @24 = (@31*@23) */
  w24  = (w31*w23);
  /* #85: @30 = (@30+@24) */
  w30 += w24;
  /* #86: @24 = (@2*@29) */
  w24  = (w2*w29);
  /* #87: @30 = (@30+@24) */
  w30 += w24;
  /* #88: @24 = (@31*@28) */
  w24  = (w31*w28);
  /* #89: @30 = (@30-@24) */
  w30 -= w24;
  /* #90: @27 = (-@27) */
  w27 = (- w27 );
  /* #91: @27 = (2.*@27) */
  w27 = (2.* w27 );
  /* #92: @16 = (@16*@27) */
  w16 *= w27;
  /* #93: @30 = (@30+@16) */
  w30 += w16;
  /* #94: (@17[3] += @30) */
  for (rr=w17+3, ss=(&w30); rr!=w17+4; rr+=1) *rr += *ss++;
  /* #95: @30 = (2.*@22) */
  w30 = (2.* w22 );
  /* #96: @15 = (@30*@15) */
  w15  = (w30*w15);
  /* #97: @14 = (@26*@14) */
  w14  = (w26*w14);
  /* #98: @15 = (@15+@14) */
  w15 += w14;
  /* #99: @13 = (@31*@13) */
  w13  = (w31*w13);
  /* #100: @15 = (@15-@13) */
  w15 -= w13;
  /* #101: @25 = (@26*@25) */
  w25  = (w26*w25);
  /* #102: @15 = (@15+@25) */
  w15 += w25;
  /* #103: @23 = (@2*@23) */
  w23  = (w2*w23);
  /* #104: @15 = (@15+@23) */
  w15 += w23;
  /* #105: @29 = (@31*@29) */
  w29  = (w31*w29);
  /* #106: @15 = (@15+@29) */
  w15 += w29;
  /* #107: @28 = (@2*@28) */
  w28  = (w2*w28);
  /* #108: @15 = (@15+@28) */
  w15 += w28;
  /* #109: @30 = (@30*@27) */
  w30 *= w27;
  /* #110: @15 = (@15+@30) */
  w15 += w30;
  /* #111: (@17[2] += @15) */
  for (rr=w17+2, ss=(&w15); rr!=w17+3; rr+=1) *rr += *ss++;
  /* #112: @18 = input[0][3] */
  casadi_copy(arg[0] ? arg[0]+10 : 0, 3, w18);
  /* #113: @15 = @18[0] */
  for (rr=(&w15), ss=w18+0; ss!=w18+1; ss+=1) *rr++ = *ss;
  /* #114: @30 = (-@15) */
  w30 = (- w15 );
  /* #115: @27 = @18[1] */
  for (rr=(&w27), ss=w18+1; ss!=w18+2; ss+=1) *rr++ = *ss;
  /* #116: @28 = (-@27) */
  w28 = (- w27 );
  /* #117: @29 = @18[2] */
  for (rr=(&w29), ss=w18+2; ss!=w18+3; ss+=1) *rr++ = *ss;
  /* #118: @23 = (-@29) */
  w23 = (- w29 );
  /* #119: @33 = horzcat(@11, @30, @28, @23) */
  rr=w33;
  *rr++ = w11;
  *rr++ = w30;
  *rr++ = w28;
  *rr++ = w23;
  /* #120: @33 = @33' */
  /* #121: @34 = horzcat(@15, @11, @29, @28) */
  rr=w34;
  *rr++ = w15;
  *rr++ = w11;
  *rr++ = w29;
  *rr++ = w28;
  /* #122: @34 = @34' */
  /* #123: @35 = horzcat(@27, @23, @11, @15) */
  rr=w35;
  *rr++ = w27;
  *rr++ = w23;
  *rr++ = w11;
  *rr++ = w15;
  /* #124: @35 = @35' */
  /* #125: @36 = horzcat(@29, @27, @30, @11) */
  rr=w36;
  *rr++ = w29;
  *rr++ = w27;
  *rr++ = w30;
  *rr++ = w11;
  /* #126: @36 = @36' */
  /* #127: @37 = horzcat(@33, @34, @35, @36) */
  rr=w37;
  for (i=0, cs=w33; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w34; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w35; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w36; i<4; ++i) *rr++ = *cs++;
  /* #128: @30 = 0.5 */
  w30 = 5.0000000000000000e-01;
  /* #129: @6 = (@30*@6) */
  for (i=0, rr=w6, cs=w6; i<4; ++i) (*rr++)  = (w30*(*cs++));
  /* #130: @33 = mac(@37,@6,@0) */
  casadi_copy(w0, 4, w33);
  for (i=0, rr=w33; i<1; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w37+j, tt=w6+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #131: @17 = (@17+@33) */
  for (i=0, rr=w17, cs=w33; i<4; ++i) (*rr++) += (*cs++);
  /* #132: output[0][0] = @17 */
  casadi_copy(w17, 4, res[0]);
  /* #133: output[0][1] = @5 */
  if (res[0]) casadi_copy(w5, 3, res[0]+4);
  /* #134: @5 = zeros(3x1) */
  casadi_clear(w5, 3);
  /* #135: @30 = -0.000102925 */
  w30 = -1.0292480999999840e-04;
  /* #136: @11 = (@30*@15) */
  w11  = (w30*w15);
  /* #137: @23 = 23.7025 */
  w23 = 2.3702544317099829e+01;
  /* #138: @23 = (@23*@10) */
  w23 *= w10;
  /* #139: @11 = (@11*@23) */
  w11 *= w23;
  /* #140: @18 = @5; (@18[1] += @11) */
  casadi_copy(w5, 3, w18);
  for (rr=w18+1, ss=(&w11); rr!=w18+2; rr+=1) *rr += *ss++;
  /* #141: @11 = (@27*@23) */
  w11  = (w27*w23);
  /* #142: @30 = (@30*@11) */
  w30 *= w11;
  /* #143: (@18[0] += @30) */
  for (rr=w18+0, ss=(&w30); rr!=w18+1; rr+=1) *rr += *ss++;
  /* #144: @30 = 0.0185744 */
  w30 = 1.8574378599999997e-02;
  /* #145: @11 = (@30*@29) */
  w11  = (w30*w29);
  /* #146: @10 = 42.1619 */
  w10 = 4.2161875870262492e+01;
  /* #147: @10 = (@10*@9) */
  w10 *= w9;
  /* #148: @11 = (@11*@10) */
  w11 *= w10;
  /* #149: (@18[0] += @11) */
  for (rr=w18+0, ss=(&w11); rr!=w18+1; rr+=1) *rr += *ss++;
  /* #150: @15 = (@15*@10) */
  w15 *= w10;
  /* #151: @30 = (@30*@15) */
  w30 *= w15;
  /* #152: (@18[2] += @30) */
  for (rr=w18+2, ss=(&w30); rr!=w18+3; rr+=1) *rr += *ss++;
  /* #153: @30 = -0.0184715 */
  w30 = -1.8471453789999998e-02;
  /* #154: @27 = (@30*@27) */
  w27  = (w30*w27);
  /* #155: @15 = 42.3456 */
  w15 = 4.2345634882548048e+01;
  /* #156: @15 = (@15*@8) */
  w15 *= w8;
  /* #157: @27 = (@27*@15) */
  w27 *= w15;
  /* #158: (@18[2] += @27) */
  for (rr=w18+2, ss=(&w27); rr!=w18+3; rr+=1) *rr += *ss++;
  /* #159: @29 = (@29*@15) */
  w29 *= w15;
  /* #160: @30 = (@30*@29) */
  w30 *= w29;
  /* #161: (@18[1] += @30) */
  for (rr=w18+1, ss=(&w30); rr!=w18+2; rr+=1) *rr += *ss++;
  /* #162: @37 = zeros(4x4) */
  casadi_clear(w37, 16);
  /* #163: @1 = @1' */
  /* #164: @37 = mac(@6,@1,@37) */
  for (i=0, rr=w37; i<4; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w6+j, tt=w1+i*1; k<1; ++k) *rr += ss[k*4]**tt++;
  /* #165: @38 = @37' */
  for (i=0, rr=w38, cs=w37; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #166: {@6, @1, @17, @33} = horzsplit(@38) */
  casadi_copy(w38, 4, w6);
  casadi_copy(w38+4, 4, w1);
  casadi_copy(w38+8, 4, w17);
  casadi_copy(w38+12, 4, w33);
  /* #167: @33 = @33' */
  /* #168: {@30, @29, @27, NULL} = horzsplit(@33) */
  w30 = w33[0];
  w29 = w33[1];
  w27 = w33[2];
  /* #169: @27 = (-@27) */
  w27 = (- w27 );
  /* #170: (@18[0] += @27) */
  for (rr=w18+0, ss=(&w27); rr!=w18+1; rr+=1) *rr += *ss++;
  /* #171: (@18[1] += @29) */
  for (rr=w18+1, ss=(&w29); rr!=w18+2; rr+=1) *rr += *ss++;
  /* #172: (@18[2] += @30) */
  for (rr=w18+2, ss=(&w30); rr!=w18+3; rr+=1) *rr += *ss++;
  /* #173: @17 = @17' */
  /* #174: {@30, @29, NULL, @27} = horzsplit(@17) */
  w30 = w17[0];
  w29 = w17[1];
  w27 = w17[3];
  /* #175: (@18[0] += @27) */
  for (rr=w18+0, ss=(&w27); rr!=w18+1; rr+=1) *rr += *ss++;
  /* #176: @29 = (-@29) */
  w29 = (- w29 );
  /* #177: (@18[2] += @29) */
  for (rr=w18+2, ss=(&w29); rr!=w18+3; rr+=1) *rr += *ss++;
  /* #178: (@18[1] += @30) */
  for (rr=w18+1, ss=(&w30); rr!=w18+2; rr+=1) *rr += *ss++;
  /* #179: @1 = @1' */
  /* #180: {@30, NULL, @29, @27} = horzsplit(@1) */
  w30 = w1[0];
  w29 = w1[2];
  w27 = w1[3];
  /* #181: @27 = (-@27) */
  w27 = (- w27 );
  /* #182: (@18[1] += @27) */
  for (rr=w18+1, ss=(&w27); rr!=w18+2; rr+=1) *rr += *ss++;
  /* #183: (@18[2] += @29) */
  for (rr=w18+2, ss=(&w29); rr!=w18+3; rr+=1) *rr += *ss++;
  /* #184: (@18[0] += @30) */
  for (rr=w18+0, ss=(&w30); rr!=w18+1; rr+=1) *rr += *ss++;
  /* #185: @6 = @6' */
  /* #186: {NULL, @30, @29, @27} = horzsplit(@6) */
  w30 = w6[1];
  w29 = w6[2];
  w27 = w6[3];
  /* #187: @27 = (-@27) */
  w27 = (- w27 );
  /* #188: (@18[2] += @27) */
  for (rr=w18+2, ss=(&w27); rr!=w18+3; rr+=1) *rr += *ss++;
  /* #189: @29 = (-@29) */
  w29 = (- w29 );
  /* #190: (@18[1] += @29) */
  for (rr=w18+1, ss=(&w29); rr!=w18+2; rr+=1) *rr += *ss++;
  /* #191: @30 = (-@30) */
  w30 = (- w30 );
  /* #192: (@18[0] += @30) */
  for (rr=w18+0, ss=(&w30); rr!=w18+1; rr+=1) *rr += *ss++;
  /* #193: output[0][2] = @18 */
  if (res[0]) casadi_copy(w18, 3, res[0]+7);
  /* #194: @6 = [-0.016, 0.016, -0.016, 0.016] */
  casadi_copy(casadi_c0, 4, w6);
  /* #195: @6 = @6' */
  /* #196: @6 = (@23*@6) */
  for (i=0, rr=w6, cs=w6; i<4; ++i) (*rr++)  = (w23*(*cs++));
  /* #197: @6 = @6' */
  /* #198: @1 = [0.173948, -0.173948, -0.173948, 0.173948] */
  casadi_copy(casadi_c1, 4, w1);
  /* #199: @1 = @1' */
  /* #200: @1 = (@10*@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) (*rr++)  = (w10*(*cs++));
  /* #201: @1 = (-@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) *rr++ = (- *cs++ );
  /* #202: @1 = @1' */
  /* #203: @6 = (@6+@1) */
  for (i=0, rr=w6, cs=w1; i<4; ++i) (*rr++) += (*cs++);
  /* #204: @1 = [-0.173948, -0.173948, 0.173948, 0.173948] */
  casadi_copy(casadi_c2, 4, w1);
  /* #205: @1 = @1' */
  /* #206: @1 = (@15*@1) */
  for (i=0, rr=w1, cs=w1; i<4; ++i) (*rr++)  = (w15*(*cs++));
  /* #207: @1 = @1' */
  /* #208: @6 = (@6+@1) */
  for (i=0, rr=w6, cs=w1; i<4; ++i) (*rr++) += (*cs++);
  /* #209: @6 = (@12*@6) */
  for (i=0, rr=w6, cs=w6; i<4; ++i) (*rr++)  = (w12*(*cs++));
  /* #210: @15 = 0.484496 */
  w15 = 4.8449612403100772e-01;
  /* #211: @10 = 1 */
  w10 = 1.;
  /* #212: @23 = sq(@22) */
  w23 = casadi_sq( w22 );
  /* #213: @30 = sq(@26) */
  w30 = casadi_sq( w26 );
  /* #214: @29 = (@23+@30) */
  w29  = (w23+w30);
  /* #215: @29 = (2.*@29) */
  w29 = (2.* w29 );
  /* #216: @29 = (@10-@29) */
  w29  = (w10-w29);
  /* #217: @27 = (@2*@22) */
  w27  = (w2*w22);
  /* #218: @8 = (@31*@26) */
  w8  = (w31*w26);
  /* #219: @11 = (@27-@8) */
  w11  = (w27-w8);
  /* #220: @11 = (2.*@11) */
  w11 = (2.* w11 );
  /* #221: @9 = (@2*@26) */
  w9  = (w2*w26);
  /* #222: @28 = (@31*@22) */
  w28  = (w31*w22);
  /* #223: @25 = (@9+@28) */
  w25  = (w9+w28);
  /* #224: @25 = (2.*@25) */
  w25 = (2.* w25 );
  /* #225: @18 = horzcat(@29, @11, @25) */
  rr=w18;
  *rr++ = w29;
  *rr++ = w11;
  *rr++ = w25;
  /* #226: @18 = @18' */
  /* #227: @27 = (@27+@8) */
  w27 += w8;
  /* #228: @27 = (2.*@27) */
  w27 = (2.* w27 );
  /* #229: @8 = sq(@2) */
  w8 = casadi_sq( w2 );
  /* #230: @30 = (@8+@30) */
  w30  = (w8+w30);
  /* #231: @30 = (2.*@30) */
  w30 = (2.* w30 );
  /* #232: @30 = (@10-@30) */
  w30  = (w10-w30);
  /* #233: @22 = (@22*@26) */
  w22 *= w26;
  /* #234: @31 = (@31*@2) */
  w31 *= w2;
  /* #235: @2 = (@22-@31) */
  w2  = (w22-w31);
  /* #236: @2 = (2.*@2) */
  w2 = (2.* w2 );
  /* #237: @20 = horzcat(@27, @30, @2) */
  rr=w20;
  *rr++ = w27;
  *rr++ = w30;
  *rr++ = w2;
  /* #238: @20 = @20' */
  /* #239: @9 = (@9-@28) */
  w9 -= w28;
  /* #240: @9 = (2.*@9) */
  w9 = (2.* w9 );
  /* #241: @22 = (@22+@31) */
  w22 += w31;
  /* #242: @22 = (2.*@22) */
  w22 = (2.* w22 );
  /* #243: @8 = (@8+@23) */
  w8 += w23;
  /* #244: @8 = (2.*@8) */
  w8 = (2.* w8 );
  /* #245: @10 = (@10-@8) */
  w10 -= w8;
  /* #246: @21 = horzcat(@9, @22, @10) */
  rr=w21;
  *rr++ = w9;
  *rr++ = w22;
  *rr++ = w10;
  /* #247: @21 = @21' */
  /* #248: @19 = horzcat(@18, @20, @21) */
  rr=w19;
  for (i=0, cs=w18; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w20; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w21; i<3; ++i) *rr++ = *cs++;
  /* #249: @5 = mac(@19,@7,@5) */
  for (i=0, rr=w5; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w19+j, tt=w7+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #250: @5 = (@15*@5) */
  for (i=0, rr=w5, cs=w5; i<3; ++i) (*rr++)  = (w15*(*cs++));
  /* #251: {NULL, NULL, @15} = vertsplit(@5) */
  w15 = w5[2];
  /* #252: (@0[3] += @15) */
  for (rr=w0+3, ss=(&w15); rr!=w0+4; rr+=1) *rr += *ss++;
  /* #253: (@0[2] += @15) */
  for (rr=w0+2, ss=(&w15); rr!=w0+3; rr+=1) *rr += *ss++;
  /* #254: (@0[1] += @15) */
  for (rr=w0+1, ss=(&w15); rr!=w0+2; rr+=1) *rr += *ss++;
  /* #255: (@0[0] += @15) */
  for (rr=w0+0, ss=(&w15); rr!=w0+1; rr+=1) *rr += *ss++;
  /* #256: @0 = (@12*@0) */
  for (i=0, rr=w0, cs=w0; i<4; ++i) (*rr++)  = (w12*(*cs++));
  /* #257: @6 = (@6+@0) */
  for (i=0, rr=w6, cs=w0; i<4; ++i) (*rr++) += (*cs++);
  /* #258: {@12, @15, @9, @22} = vertsplit(@6) */
  w12 = w6[0];
  w15 = w6[1];
  w9 = w6[2];
  w22 = w6[3];
  /* #259: output[0][3] = @12 */
  if (res[0]) res[0][10] = w12;
  /* #260: output[0][4] = @15 */
  if (res[0]) res[0][11] = w15;
  /* #261: output[0][5] = @9 */
  if (res[0]) res[0][12] = w9;
  /* #262: output[0][6] = @22 */
  if (res[0]) res[0][13] = w22;
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_adj(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_adj_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_adj_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_adj_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_adj_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_adj_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_adj_incref(void) {
}

CASADI_SYMBOL_EXPORT void quad_expl_vde_adj_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quad_expl_vde_adj_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int quad_expl_vde_adj_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real quad_expl_vde_adj_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quad_expl_vde_adj_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quad_expl_vde_adj_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quad_expl_vde_adj_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quad_expl_vde_adj_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_adj_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 8;
  if (sz_res) *sz_res = 7;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 135;
  return 0;
}

CASADI_SYMBOL_EXPORT int quad_expl_vde_adj_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 8*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 7*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 135*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
