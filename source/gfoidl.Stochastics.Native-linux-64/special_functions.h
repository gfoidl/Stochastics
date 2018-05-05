#pragma once
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif
    double gaussian_error_function(double x);
    double gaussian_error_function_complementary(double x);

    void gaussian_error_function_vector(double* values, double* result, const int n);
    void gaussian_error_function_complementary_vector(double* values, double* result, const int n);
#ifdef __cplusplus
}
#endif
