/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: conv2.c
 *
 * MATLAB Coder version            : 5.5
 * C/C++ source code generated on  : 17-Jan-2023 01:28:46
 */

/* Include Files */
#include "conv2.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const double a[960]
 *                const float b[96]
 *                float c[55]
 * Return Type  : void
 */
void b_conv2(const double a[960], const float b[96], float c[55])
{
  float cj;
  int ib;
  int j;
  int jb;
  for (j = 0; j < 55; j++) {
    cj = 0.0F;
    for (jb = 0; jb < 6; jb++) {
      for (ib = 0; ib < 16; ib++) {
        cj += b[(((5 - jb) << 4) - ib) + 15] * (float)a[ib + ((j + jb) << 4)];
      }
    }
    c[j] = cj;
  }
}

/*
 * Arguments    : const float a[756]
 *                const float b[36]
 *                float c[121]
 * Return Type  : void
 */
void conv2(const float a[756], const float b[36], float c[121])
{
  float cj;
  int ib;
  int j;
  int jb;
  for (j = 0; j < 121; j++) {
    cj = 0.0F;
    for (jb = 0; jb < 6; jb++) {
      for (ib = 0; ib < 6; ib++) {
        cj += b[(6 * (5 - jb) - ib) + 5] * a[ib + 6 * (j + jb)];
      }
    }
    c[j] = cj;
  }
}

/*
 * File trailer for conv2.c
 *
 * [EOF]
 */
