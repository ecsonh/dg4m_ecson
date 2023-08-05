/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: rand.c
 *
 * MATLAB Coder version            : 5.5
 * C/C++ source code generated on  : 17-Jan-2023 01:28:46
 */

/* Include Files */
#include "rand.h"
#include "CNN1_Predict_data.h"
#include "eml_rand_mt19937ar.h"
#include "eml_rand_shr3cong.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : float r[16]
 * Return Type  : void
 */
void b_rand(float r[16])
{
  float b_r;
  unsigned int a;
  unsigned int b;
  int hi;
  int k;
  if (method == 4U) {
    for (k = 0; k < 16; k++) {
      hi = (int)(state / 127773U);
      a = 16807U * (state - (unsigned int)hi * 127773U);
      b = 2836U * (unsigned int)hi;
      if (a < b) {
        state = ~(b - a) & 2147483647U;
      } else {
        state = a - b;
      }
      b_r = (float)state * 4.65661287E-10F;
      if (b_r == 0.0F) {
        b_r = 1.17549435E-38F;
      } else if (b_r == 1.0F) {
        b_r = 0.99999994F;
      }
      r[k] = b_r;
    }
  } else if (method == 5U) {
    for (k = 0; k < 16; k++) {
      r[k] = eml_rand_shr3cong(b_state);
    }
  } else {
    for (k = 0; k < 16; k++) {
      r[k] = eml_rand_mt19937ar(c_state);
    }
  }
}

/*
 * Arguments    : float r[32]
 * Return Type  : void
 */
void c_rand(float r[32])
{
  float b_r;
  unsigned int a;
  unsigned int b;
  int hi;
  int k;
  if (method == 4U) {
    for (k = 0; k < 32; k++) {
      hi = (int)(state / 127773U);
      a = 16807U * (state - (unsigned int)hi * 127773U);
      b = 2836U * (unsigned int)hi;
      if (a < b) {
        state = ~(b - a) & 2147483647U;
      } else {
        state = a - b;
      }
      b_r = (float)state * 4.65661287E-10F;
      if (b_r == 0.0F) {
        b_r = 1.17549435E-38F;
      } else if (b_r == 1.0F) {
        b_r = 0.99999994F;
      }
      r[k] = b_r;
    }
  } else if (method == 5U) {
    for (k = 0; k < 32; k++) {
      r[k] = eml_rand_shr3cong(b_state);
    }
  } else {
    for (k = 0; k < 32; k++) {
      r[k] = eml_rand_mt19937ar(c_state);
    }
  }
}

/*
 * File trailer for rand.c
 *
 * [EOF]
 */
