#include <cmath>
#include "special_functions.h"
//-----------------------------------------------------------------------------
double gaussian_error_function(double x)
{
    return std::erf(x);
}
//-----------------------------------------------------------------------------
double gaussian_error_function_complementary(double x)
{
    return std::erfc(x);
}
//-----------------------------------------------------------------------------
void gaussian_error_function_vector(double* values, double* result, const int n)
{
    for (int i = 0; i < n; ++i)
        result[i] = std::erf(values[i]);
}
//-----------------------------------------------------------------------------
void gaussian_error_function_complementary_vector(double* values, double* result, const int n)
{
    for (int i = 0; i < n; ++i)
        result[i] = std::erfc(values[i]);
}
