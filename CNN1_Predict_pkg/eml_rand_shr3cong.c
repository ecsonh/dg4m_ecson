/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: eml_rand_shr3cong.c
 *
 * MATLAB Coder version            : 5.5
 * C/C++ source code generated on  : 17-Jan-2023 01:28:46
 */

/* Include Files */
#include "eml_rand_shr3cong.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : unsigned int d_state[2]
 * Return Type  : float
 */
float eml_rand_shr3cong(unsigned int d_state[2])
{
  float r;
  unsigned int u;
  unsigned int u1;
  u = 69069U * d_state[0] + 1234567U;
  u1 = d_state[1] ^ d_state[1] << 13;
  u1 ^= u1 >> 17;
  u1 ^= u1 << 5;
  d_state[0] = u;
  d_state[1] = u1;
  r = (float)(u + u1) * 2.32830644E-10F;
  if (r == 0.0F) {
    r = 1.17549435E-38F;
  } else if (r == 1.0F) {
    r = 0.99999994F;
  }
  return r;
}

/*
 * File trailer for eml_rand_shr3cong.c
 *
 * [EOF]
 */
