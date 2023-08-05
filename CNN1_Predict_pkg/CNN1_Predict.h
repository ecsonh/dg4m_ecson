/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: CNN1_Predict.h
 *
 * MATLAB Coder version            : 5.5
 * C/C++ source code generated on  : 17-Jan-2023 01:28:46
 */

#ifndef CNN1_PREDICT_H
#define CNN1_PREDICT_H

/* Include Files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void CNN1_Predict(const float data[756], const float weights1[576],
                         const float weights2[3072],
                         const float weights3[10368], float output[12]);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for CNN1_Predict.h
 *
 * [EOF]
 */
