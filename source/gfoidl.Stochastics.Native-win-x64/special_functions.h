#pragma once
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif
    DLL_API double gaussian_error_function(double x);
    DLL_API double gaussian_error_function_complementary(double x);

    DLL_API void gaussian_error_function_vector(double* values, double* result, const int n);
    DLL_API void gaussian_error_function_complementary_vector(double* values, double* result, const int n);
#ifdef __cplusplus
}
#endif
