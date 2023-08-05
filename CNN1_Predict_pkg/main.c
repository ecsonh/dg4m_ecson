/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: main.c
 *
 * MATLAB Coder version            : 5.5
 * C/C++ source code generated on  : 17-Jan-2023 01:28:46
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/

/* Include Files */
#include "main.h"
#include "CNN1_Predict.h"
#include "CNN1_Predict_terminate.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static void argInit_16x6x32_real32_T(float result[3072]);

static void argInit_6x126_real32_T(float result[756]);

static void argInit_6x6x16_real32_T(float result[576]);

static void argInit_864x12_real32_T(float result[10368]);

static float argInit_real32_T(void);

/* Function Definitions */
/*
 * Arguments    : float result[3072]
 * Return Type  : void
 */
static void argInit_16x6x32_real32_T(float result[3072])
{
  int idx0;
  int idx1;
  int idx2;
  /* Loop over the array to initialize each element. */
  for (idx1 = 0; idx1 < 6; idx1++) {
    for (idx2 = 0; idx2 < 32; idx2++) {
      for (idx0 = 0; idx0 < 16; idx0++) {
        /* Set the value of the array element.
Change this value to the value that the application requires. */
        result[(idx0 + (idx1 << 4)) + 96 * idx2] = argInit_real32_T();
      }
    }
  }
}

/*
 * Arguments    : float result[756]
 * Return Type  : void
 */
static void argInit_6x126_real32_T(float result[756])
{
  int i;
  /* Loop over the array to initialize each element. */
  for (i = 0; i < 756; i++) {
    /* Set the value of the array element.
Change this value to the value that the application requires. */
    result[i] = argInit_real32_T();
  }
}

/*
 * Arguments    : float result[576]
 * Return Type  : void
 */
static void argInit_6x6x16_real32_T(float result[576])
{
  int idx0;
  int idx1;
  int idx2;
  /* Loop over the array to initialize each element. */
  for (idx1 = 0; idx1 < 6; idx1++) {
    for (idx2 = 0; idx2 < 16; idx2++) {
      for (idx0 = 0; idx0 < 6; idx0++) {
        /* Set the value of the array element.
Change this value to the value that the application requires. */
        result[(idx0 + 6 * idx1) + 36 * idx2] = argInit_real32_T();
      }
    }
  }
}

/*
 * Arguments    : float result[10368]
 * Return Type  : void
 */
static void argInit_864x12_real32_T(float result[10368])
{
  int i;
  /* Loop over the array to initialize each element. */
  for (i = 0; i < 10368; i++) {
    /* Set the value of the array element.
Change this value to the value that the application requires. */
    result[i] = argInit_real32_T();
  }
}

/*
 * Arguments    : void
 * Return Type  : float
 */
static float argInit_real32_T(void)
{
  return 0.0F;
}

/*
 * Arguments    : int argc
 *                char **argv
 * Return Type  : int
 */
int main(int argc, char **argv)
{
  (void)argc;
  (void)argv;
  /* The initialize function is being called automatically from your entry-point
   * function. So, a call to initialize is not included here. */
  /* Invoke the entry-point functions.
You can call entry-point functions multiple times. */
  main_CNN1_Predict();
  /* Terminate the application.
You do not need to do this more than one time. */
  CNN1_Predict_terminate();
  return 0;
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void main_CNN1_Predict(void)
{
  static float fv3[10368];
  static float fv2[3072];
  static float fv[756];
  static float fv1[576];
  float output[12];
  /* Initialize function 'CNN1_Predict' input arguments. */
  /* Initialize function input argument 'data'. */
  /* Initialize function input argument 'weights1'. */
  /* Initialize function input argument 'weights2'. */
  /* Initialize function input argument 'weights3'. */
  /* Call the entry-point 'CNN1_Predict'. */
  argInit_6x126_real32_T(fv);
  argInit_6x6x16_real32_T(fv1);
  argInit_16x6x32_real32_T(fv2);
  argInit_864x12_real32_T(fv3);
  CNN1_Predict(fv, fv1, fv2, fv3, output);
}

/*
 * File trailer for main.c
 *
 * [EOF]
 */
