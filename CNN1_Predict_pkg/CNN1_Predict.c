/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: CNN1_Predict.c
 *
 * MATLAB Coder version            : 5.5
 * C/C++ source code generated on  : 17-Jan-2023 01:28:46
 */

/* Include Files */
#include "CNN1_Predict.h"
#include "CNN1_Predict_data.h"
#include "CNN1_Predict_initialize.h"
#include "conv2.h"
#include "rand.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * ,weights4)
 *
 * Arguments    : const float data[756]
 *                const float weights1[576]
 *                const float weights2[3072]
 *                const float weights3[10368]
 *                float output[12]
 * Return Type  : void
 */
void CNN1_Predict(const float data[756], const float weights1[576],
                  const float weights2[3072], const float weights3[10368],
                  float output[12])
{
  static double input[960];
  static double output1_1[864];
  static float output1[1936];
  static float output2[1936];
  static float output4[1760];
  static float output5[1760];
  static float b_gamma[121];
  static float fv[121];
  static float fv1[121];
  static float fv10[121];
  static float fv11[121];
  static float fv2[121];
  static float fv3[121];
  static float fv4[121];
  static float fv5[121];
  static float fv6[121];
  static float fv7[121];
  static float fv8[121];
  static float fv9[121];
  float fv12[121];
  float fv13[121];
  float fv14[121];
  float d_gamma[55];
  float fv15[55];
  float fv16[55];
  float fv17[55];
  float fv18[55];
  float fv19[55];
  float fv20[55];
  float fv21[55];
  float fv22[55];
  float fv23[55];
  float fv24[55];
  float fv25[55];
  float fv26[55];
  float fv27[55];
  float fv28[55];
  float fv29[55];
  float fv30[55];
  float fv31[55];
  float fv32[55];
  float fv33[55];
  float fv34[55];
  float fv35[55];
  float fv36[55];
  float fv37[55];
  float fv38[55];
  float fv39[55];
  float fv40[55];
  float fv41[55];
  float fv42[55];
  float fv43[55];
  float fv44[55];
  float fv45[55];
  float b_beta[32];
  float b_bias[32];
  float c_std[32];
  float e_gamma[32];
  float b_std[16];
  float beta[16];
  float bias[16];
  float c_gamma[16];
  float f;
  float f1;
  float f10;
  float f11;
  float f12;
  float f13;
  float f14;
  float f15;
  float f16;
  float f17;
  float f18;
  float f19;
  float f2;
  float f20;
  float f21;
  float f22;
  float f23;
  float f24;
  float f25;
  float f26;
  float f27;
  float f28;
  float f29;
  float f3;
  float f30;
  float f31;
  float f32;
  float f33;
  float f34;
  float f35;
  float f36;
  float f37;
  float f38;
  float f39;
  float f4;
  float f40;
  float f41;
  float f42;
  float f43;
  float f44;
  float f45;
  float f46;
  float f47;
  float f48;
  float f49;
  float f5;
  float f50;
  float f51;
  float f52;
  float f53;
  float f54;
  float f55;
  float f56;
  float f57;
  float f58;
  float f59;
  float f6;
  float f60;
  float f61;
  float f62;
  float f63;
  float f7;
  float f8;
  float f9;
  int b_i;
  int i;
  int output1_tmp;
  if (!isInitialized_CNN1_Predict) {
    CNN1_Predict_initialize();
  }
  b_rand(bias);
  conv2(data, &weights1[0], b_gamma);
  conv2(data, &weights1[36], fv);
  conv2(data, &weights1[72], fv1);
  conv2(data, &weights1[108], fv2);
  conv2(data, &weights1[144], fv3);
  conv2(data, &weights1[180], fv4);
  conv2(data, &weights1[216], fv5);
  conv2(data, &weights1[252], fv6);
  conv2(data, &weights1[288], fv7);
  conv2(data, &weights1[324], fv8);
  conv2(data, &weights1[360], fv9);
  conv2(data, &weights1[396], fv10);
  conv2(data, &weights1[432], fv11);
  conv2(data, &weights1[468], fv12);
  conv2(data, &weights1[504], fv13);
  conv2(data, &weights1[540], fv14);
  f = bias[0];
  f1 = bias[1];
  f2 = bias[2];
  f3 = bias[3];
  f4 = bias[4];
  f5 = bias[5];
  f6 = bias[6];
  f7 = bias[7];
  f8 = bias[8];
  f9 = bias[9];
  f10 = bias[10];
  f11 = bias[11];
  f12 = bias[12];
  f13 = bias[13];
  f14 = bias[14];
  f15 = bias[15];
  for (i = 0; i < 121; i++) {
    output1_tmp = i << 4;
    output1[output1_tmp] = b_gamma[i] + f;
    output1[output1_tmp + 1] = fv[i] + f1;
    output1[output1_tmp + 2] = fv1[i] + f2;
    output1[output1_tmp + 3] = fv2[i] + f3;
    output1[output1_tmp + 4] = fv3[i] + f4;
    output1[output1_tmp + 5] = fv4[i] + f5;
    output1[output1_tmp + 6] = fv5[i] + f6;
    output1[output1_tmp + 7] = fv6[i] + f7;
    output1[output1_tmp + 8] = fv7[i] + f8;
    output1[output1_tmp + 9] = fv8[i] + f9;
    output1[output1_tmp + 10] = fv9[i] + f10;
    output1[output1_tmp + 11] = fv10[i] + f11;
    output1[output1_tmp + 12] = fv11[i] + f12;
    output1[output1_tmp + 13] = fv12[i] + f13;
    output1[output1_tmp + 14] = fv13[i] + f14;
    output1[output1_tmp + 15] = fv14[i] + f15;
  }
  /*  output1 = output1(:,1:2:541); % Stride 2  */
  /*  output1(output1<0) = 0;   % relu after BN */
  b_rand(bias);
  b_rand(b_std);
  b_rand(c_gamma);
  b_rand(beta);
  for (b_i = 0; b_i < 16; b_i++) {
    for (i = 0; i < 121; i++) {
      output1_tmp = b_i + (i << 4);
      output1[output1_tmp] =
          c_gamma[b_i] * ((output1[output1_tmp] - bias[b_i]) / b_std[b_i]) +
          beta[b_i];
    }
  }
  for (b_i = 0; b_i < 1936; b_i++) {
    f = output1[b_i];
    output2[b_i] = f;
    if (f < 0.0F) {
      output2[b_i] = 0.0F;
    }
  }
  /*  Relu after BN */
  for (b_i = 0; b_i < 60; b_i++) {
    output1_tmp = (b_i << 1) + 1;
    i = (output1_tmp - 1) << 4;
    f = output2[i];
    output1_tmp <<= 4;
    f1 = output2[output1_tmp];
    f2 = output2[i + 1];
    f3 = output2[output1_tmp + 1];
    f4 = output2[i + 2];
    f5 = output2[output1_tmp + 2];
    f6 = output2[i + 3];
    f7 = output2[output1_tmp + 3];
    f8 = output2[i + 4];
    f9 = output2[output1_tmp + 4];
    f10 = output2[i + 5];
    f11 = output2[output1_tmp + 5];
    f12 = output2[i + 6];
    f13 = output2[output1_tmp + 6];
    f14 = output2[i + 7];
    f15 = output2[output1_tmp + 7];
    f16 = output2[i + 8];
    f17 = output2[output1_tmp + 8];
    f18 = output2[i + 9];
    f19 = output2[output1_tmp + 9];
    f20 = output2[i + 10];
    f21 = output2[output1_tmp + 10];
    f22 = output2[i + 11];
    f23 = output2[output1_tmp + 11];
    f24 = output2[i + 12];
    f25 = output2[output1_tmp + 12];
    f26 = output2[i + 13];
    f27 = output2[output1_tmp + 13];
    f28 = output2[i + 14];
    f29 = output2[output1_tmp + 14];
    f30 = output2[i + 15];
    f31 = output2[output1_tmp + 15];
    output1_tmp = b_i << 4;
    if ((f >= f1) || rtIsNaNF(f1)) {
      input[output1_tmp] = f;
    } else {
      input[output1_tmp] = f1;
    }
    if ((f2 >= f3) || rtIsNaNF(f3)) {
      input[output1_tmp + 1] = f2;
    } else {
      input[output1_tmp + 1] = f3;
    }
    if ((f4 >= f5) || rtIsNaNF(f5)) {
      input[output1_tmp + 2] = f4;
    } else {
      input[output1_tmp + 2] = f5;
    }
    if ((f6 >= f7) || rtIsNaNF(f7)) {
      input[output1_tmp + 3] = f6;
    } else {
      input[output1_tmp + 3] = f7;
    }
    if ((f8 >= f9) || rtIsNaNF(f9)) {
      input[output1_tmp + 4] = f8;
    } else {
      input[output1_tmp + 4] = f9;
    }
    if ((f10 >= f11) || rtIsNaNF(f11)) {
      input[output1_tmp + 5] = f10;
    } else {
      input[output1_tmp + 5] = f11;
    }
    if ((f12 >= f13) || rtIsNaNF(f13)) {
      input[output1_tmp + 6] = f12;
    } else {
      input[output1_tmp + 6] = f13;
    }
    if ((f14 >= f15) || rtIsNaNF(f15)) {
      input[output1_tmp + 7] = f14;
    } else {
      input[output1_tmp + 7] = f15;
    }
    if ((f16 >= f17) || rtIsNaNF(f17)) {
      input[output1_tmp + 8] = f16;
    } else {
      input[output1_tmp + 8] = f17;
    }
    if ((f18 >= f19) || rtIsNaNF(f19)) {
      input[output1_tmp + 9] = f18;
    } else {
      input[output1_tmp + 9] = f19;
    }
    if ((f20 >= f21) || rtIsNaNF(f21)) {
      input[output1_tmp + 10] = f20;
    } else {
      input[output1_tmp + 10] = f21;
    }
    if ((f22 >= f23) || rtIsNaNF(f23)) {
      input[output1_tmp + 11] = f22;
    } else {
      input[output1_tmp + 11] = f23;
    }
    if ((f24 >= f25) || rtIsNaNF(f25)) {
      input[output1_tmp + 12] = f24;
    } else {
      input[output1_tmp + 12] = f25;
    }
    if ((f26 >= f27) || rtIsNaNF(f27)) {
      input[output1_tmp + 13] = f26;
    } else {
      input[output1_tmp + 13] = f27;
    }
    if ((f28 >= f29) || rtIsNaNF(f29)) {
      input[output1_tmp + 14] = f28;
    } else {
      input[output1_tmp + 14] = f29;
    }
    if ((f30 >= f31) || rtIsNaNF(f31)) {
      input[output1_tmp + 15] = f30;
    } else {
      input[output1_tmp + 15] = f31;
    }
  }
  c_rand(b_bias);
  b_conv2(input, &weights2[0], d_gamma);
  b_conv2(input, &weights2[96], fv15);
  b_conv2(input, &weights2[192], fv16);
  b_conv2(input, &weights2[288], fv17);
  b_conv2(input, &weights2[384], fv18);
  b_conv2(input, &weights2[480], fv19);
  b_conv2(input, &weights2[576], fv20);
  b_conv2(input, &weights2[672], fv21);
  b_conv2(input, &weights2[768], fv22);
  b_conv2(input, &weights2[864], fv23);
  b_conv2(input, &weights2[960], fv24);
  b_conv2(input, &weights2[1056], fv25);
  b_conv2(input, &weights2[1152], fv26);
  b_conv2(input, &weights2[1248], fv27);
  b_conv2(input, &weights2[1344], fv28);
  b_conv2(input, &weights2[1440], fv29);
  b_conv2(input, &weights2[1536], fv30);
  b_conv2(input, &weights2[1632], fv31);
  b_conv2(input, &weights2[1728], fv32);
  b_conv2(input, &weights2[1824], fv33);
  b_conv2(input, &weights2[1920], fv34);
  b_conv2(input, &weights2[2016], fv35);
  b_conv2(input, &weights2[2112], fv36);
  b_conv2(input, &weights2[2208], fv37);
  b_conv2(input, &weights2[2304], fv38);
  b_conv2(input, &weights2[2400], fv39);
  b_conv2(input, &weights2[2496], fv40);
  b_conv2(input, &weights2[2592], fv41);
  b_conv2(input, &weights2[2688], fv42);
  b_conv2(input, &weights2[2784], fv43);
  b_conv2(input, &weights2[2880], fv44);
  b_conv2(input, &weights2[2976], fv45);
  f = b_bias[0];
  f1 = b_bias[1];
  f2 = b_bias[2];
  f3 = b_bias[3];
  f4 = b_bias[4];
  f5 = b_bias[5];
  f6 = b_bias[6];
  f7 = b_bias[7];
  f8 = b_bias[8];
  f9 = b_bias[9];
  f10 = b_bias[10];
  f11 = b_bias[11];
  f12 = b_bias[12];
  f13 = b_bias[13];
  f14 = b_bias[14];
  f15 = b_bias[15];
  f16 = b_bias[16];
  f17 = b_bias[17];
  f18 = b_bias[18];
  f19 = b_bias[19];
  f20 = b_bias[20];
  f21 = b_bias[21];
  f22 = b_bias[22];
  f23 = b_bias[23];
  f24 = b_bias[24];
  f25 = b_bias[25];
  f26 = b_bias[26];
  f27 = b_bias[27];
  f28 = b_bias[28];
  f29 = b_bias[29];
  f30 = b_bias[30];
  f31 = b_bias[31];
  for (i = 0; i < 55; i++) {
    output1_tmp = i << 5;
    output4[output1_tmp] = d_gamma[i] + f;
    output4[output1_tmp + 1] = fv15[i] + f1;
    output4[output1_tmp + 2] = fv16[i] + f2;
    output4[output1_tmp + 3] = fv17[i] + f3;
    output4[output1_tmp + 4] = fv18[i] + f4;
    output4[output1_tmp + 5] = fv19[i] + f5;
    output4[output1_tmp + 6] = fv20[i] + f6;
    output4[output1_tmp + 7] = fv21[i] + f7;
    output4[output1_tmp + 8] = fv22[i] + f8;
    output4[output1_tmp + 9] = fv23[i] + f9;
    output4[output1_tmp + 10] = fv24[i] + f10;
    output4[output1_tmp + 11] = fv25[i] + f11;
    output4[output1_tmp + 12] = fv26[i] + f12;
    output4[output1_tmp + 13] = fv27[i] + f13;
    output4[output1_tmp + 14] = fv28[i] + f14;
    output4[output1_tmp + 15] = fv29[i] + f15;
    output4[output1_tmp + 16] = fv30[i] + f16;
    output4[output1_tmp + 17] = fv31[i] + f17;
    output4[output1_tmp + 18] = fv32[i] + f18;
    output4[output1_tmp + 19] = fv33[i] + f19;
    output4[output1_tmp + 20] = fv34[i] + f20;
    output4[output1_tmp + 21] = fv35[i] + f21;
    output4[output1_tmp + 22] = fv36[i] + f22;
    output4[output1_tmp + 23] = fv37[i] + f23;
    output4[output1_tmp + 24] = fv38[i] + f24;
    output4[output1_tmp + 25] = fv39[i] + f25;
    output4[output1_tmp + 26] = fv40[i] + f26;
    output4[output1_tmp + 27] = fv41[i] + f27;
    output4[output1_tmp + 28] = fv42[i] + f28;
    output4[output1_tmp + 29] = fv43[i] + f29;
    output4[output1_tmp + 30] = fv44[i] + f30;
    output4[output1_tmp + 31] = fv45[i] + f31;
  }
  /*  output1(output1<0) = 0;  % relu after BN */
  c_rand(b_bias);
  c_rand(c_std);
  c_rand(e_gamma);
  c_rand(b_beta);
  for (b_i = 0; b_i < 32; b_i++) {
    for (i = 0; i < 55; i++) {
      output1_tmp = b_i + (i << 5);
      output4[output1_tmp] =
          e_gamma[b_i] * ((output4[output1_tmp] - b_bias[b_i]) / c_std[b_i]) +
          b_beta[b_i];
    }
  }
  for (b_i = 0; b_i < 1760; b_i++) {
    f = output4[b_i];
    output5[b_i] = f;
    if (f < 0.0F) {
      output5[b_i] = 0.0F;
    }
  }
  /*  relu after BN */
  /*  Relu after BN */
  for (b_i = 0; b_i < 27; b_i++) {
    output1_tmp = (b_i << 1) + 1;
    i = (output1_tmp - 1) << 5;
    f = output5[i];
    output1_tmp <<= 5;
    f1 = output5[output1_tmp];
    f2 = output5[i + 1];
    f3 = output5[output1_tmp + 1];
    f4 = output5[i + 2];
    f5 = output5[output1_tmp + 2];
    f6 = output5[i + 3];
    f7 = output5[output1_tmp + 3];
    f8 = output5[i + 4];
    f9 = output5[output1_tmp + 4];
    f10 = output5[i + 5];
    f11 = output5[output1_tmp + 5];
    f12 = output5[i + 6];
    f13 = output5[output1_tmp + 6];
    f14 = output5[i + 7];
    f15 = output5[output1_tmp + 7];
    f16 = output5[i + 8];
    f17 = output5[output1_tmp + 8];
    f18 = output5[i + 9];
    f19 = output5[output1_tmp + 9];
    f20 = output5[i + 10];
    f21 = output5[output1_tmp + 10];
    f22 = output5[i + 11];
    f23 = output5[output1_tmp + 11];
    f24 = output5[i + 12];
    f25 = output5[output1_tmp + 12];
    f26 = output5[i + 13];
    f27 = output5[output1_tmp + 13];
    f28 = output5[i + 14];
    f29 = output5[output1_tmp + 14];
    f30 = output5[i + 15];
    f31 = output5[output1_tmp + 15];
    f32 = output5[i + 16];
    f33 = output5[output1_tmp + 16];
    f34 = output5[i + 17];
    f35 = output5[output1_tmp + 17];
    f36 = output5[i + 18];
    f37 = output5[output1_tmp + 18];
    f38 = output5[i + 19];
    f39 = output5[output1_tmp + 19];
    f40 = output5[i + 20];
    f41 = output5[output1_tmp + 20];
    f42 = output5[i + 21];
    f43 = output5[output1_tmp + 21];
    f44 = output5[i + 22];
    f45 = output5[output1_tmp + 22];
    f46 = output5[i + 23];
    f47 = output5[output1_tmp + 23];
    f48 = output5[i + 24];
    f49 = output5[output1_tmp + 24];
    f50 = output5[i + 25];
    f51 = output5[output1_tmp + 25];
    f52 = output5[i + 26];
    f53 = output5[output1_tmp + 26];
    f54 = output5[i + 27];
    f55 = output5[output1_tmp + 27];
    f56 = output5[i + 28];
    f57 = output5[output1_tmp + 28];
    f58 = output5[i + 29];
    f59 = output5[output1_tmp + 29];
    f60 = output5[i + 30];
    f61 = output5[output1_tmp + 30];
    f62 = output5[i + 31];
    f63 = output5[output1_tmp + 31];
    if ((f >= f1) || rtIsNaNF(f1)) {
      output1_1[b_i] = f;
    } else {
      output1_1[b_i] = f1;
    }
    if ((f2 >= f3) || rtIsNaNF(f3)) {
      output1_1[b_i + 27] = f2;
    } else {
      output1_1[b_i + 27] = f3;
    }
    if ((f4 >= f5) || rtIsNaNF(f5)) {
      output1_1[b_i + 54] = f4;
    } else {
      output1_1[b_i + 54] = f5;
    }
    if ((f6 >= f7) || rtIsNaNF(f7)) {
      output1_1[b_i + 81] = f6;
    } else {
      output1_1[b_i + 81] = f7;
    }
    if ((f8 >= f9) || rtIsNaNF(f9)) {
      output1_1[b_i + 108] = f8;
    } else {
      output1_1[b_i + 108] = f9;
    }
    if ((f10 >= f11) || rtIsNaNF(f11)) {
      output1_1[b_i + 135] = f10;
    } else {
      output1_1[b_i + 135] = f11;
    }
    if ((f12 >= f13) || rtIsNaNF(f13)) {
      output1_1[b_i + 162] = f12;
    } else {
      output1_1[b_i + 162] = f13;
    }
    if ((f14 >= f15) || rtIsNaNF(f15)) {
      output1_1[b_i + 189] = f14;
    } else {
      output1_1[b_i + 189] = f15;
    }
    if ((f16 >= f17) || rtIsNaNF(f17)) {
      output1_1[b_i + 216] = f16;
    } else {
      output1_1[b_i + 216] = f17;
    }
    if ((f18 >= f19) || rtIsNaNF(f19)) {
      output1_1[b_i + 243] = f18;
    } else {
      output1_1[b_i + 243] = f19;
    }
    if ((f20 >= f21) || rtIsNaNF(f21)) {
      output1_1[b_i + 270] = f20;
    } else {
      output1_1[b_i + 270] = f21;
    }
    if ((f22 >= f23) || rtIsNaNF(f23)) {
      output1_1[b_i + 297] = f22;
    } else {
      output1_1[b_i + 297] = f23;
    }
    if ((f24 >= f25) || rtIsNaNF(f25)) {
      output1_1[b_i + 324] = f24;
    } else {
      output1_1[b_i + 324] = f25;
    }
    if ((f26 >= f27) || rtIsNaNF(f27)) {
      output1_1[b_i + 351] = f26;
    } else {
      output1_1[b_i + 351] = f27;
    }
    if ((f28 >= f29) || rtIsNaNF(f29)) {
      output1_1[b_i + 378] = f28;
    } else {
      output1_1[b_i + 378] = f29;
    }
    if ((f30 >= f31) || rtIsNaNF(f31)) {
      output1_1[b_i + 405] = f30;
    } else {
      output1_1[b_i + 405] = f31;
    }
    if ((f32 >= f33) || rtIsNaNF(f33)) {
      output1_1[b_i + 432] = f32;
    } else {
      output1_1[b_i + 432] = f33;
    }
    if ((f34 >= f35) || rtIsNaNF(f35)) {
      output1_1[b_i + 459] = f34;
    } else {
      output1_1[b_i + 459] = f35;
    }
    if ((f36 >= f37) || rtIsNaNF(f37)) {
      output1_1[b_i + 486] = f36;
    } else {
      output1_1[b_i + 486] = f37;
    }
    if ((f38 >= f39) || rtIsNaNF(f39)) {
      output1_1[b_i + 513] = f38;
    } else {
      output1_1[b_i + 513] = f39;
    }
    if ((f40 >= f41) || rtIsNaNF(f41)) {
      output1_1[b_i + 540] = f40;
    } else {
      output1_1[b_i + 540] = f41;
    }
    if ((f42 >= f43) || rtIsNaNF(f43)) {
      output1_1[b_i + 567] = f42;
    } else {
      output1_1[b_i + 567] = f43;
    }
    if ((f44 >= f45) || rtIsNaNF(f45)) {
      output1_1[b_i + 594] = f44;
    } else {
      output1_1[b_i + 594] = f45;
    }
    if ((f46 >= f47) || rtIsNaNF(f47)) {
      output1_1[b_i + 621] = f46;
    } else {
      output1_1[b_i + 621] = f47;
    }
    if ((f48 >= f49) || rtIsNaNF(f49)) {
      output1_1[b_i + 648] = f48;
    } else {
      output1_1[b_i + 648] = f49;
    }
    if ((f50 >= f51) || rtIsNaNF(f51)) {
      output1_1[b_i + 675] = f50;
    } else {
      output1_1[b_i + 675] = f51;
    }
    if ((f52 >= f53) || rtIsNaNF(f53)) {
      output1_1[b_i + 702] = f52;
    } else {
      output1_1[b_i + 702] = f53;
    }
    if ((f54 >= f55) || rtIsNaNF(f55)) {
      output1_1[b_i + 729] = f54;
    } else {
      output1_1[b_i + 729] = f55;
    }
    if ((f56 >= f57) || rtIsNaNF(f57)) {
      output1_1[b_i + 756] = f56;
    } else {
      output1_1[b_i + 756] = f57;
    }
    if ((f58 >= f59) || rtIsNaNF(f59)) {
      output1_1[b_i + 783] = f58;
    } else {
      output1_1[b_i + 783] = f59;
    }
    if ((f60 >= f61) || rtIsNaNF(f61)) {
      output1_1[b_i + 810] = f60;
    } else {
      output1_1[b_i + 810] = f61;
    }
    if ((f62 >= f63) || rtIsNaNF(f63)) {
      output1_1[b_i + 837] = f62;
    } else {
      output1_1[b_i + 837] = f63;
    }
  }
  for (i = 0; i < 12; i++) {
    f = 0.0F;
    for (output1_tmp = 0; output1_tmp < 864; output1_tmp++) {
      f += (float)output1_1[output1_tmp] * weights3[output1_tmp + 864 * i];
    }
    output[i] = f + 0.9769F;
  }
  /*  Relu after Dense1 */
  /*  output9 = CNN_Second_Dense(output8,weights4); */
  /*  pred_test = Sigmoid(output8); */
  /*  output = pred_test; */
}

/*
 * File trailer for CNN1_Predict.c
 *
 * [EOF]
 */
