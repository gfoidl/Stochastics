#include "stdafx.h"
#include "special_functions.h"
//-----------------------------------------------------------------------------
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
//-----------------------------------------------------------------------------
namespace gfoidlStochasticsNativewinx64Tests
{
    TEST_CLASS(special_functions_Tests)
    {
    private:
        static double _input[];
        static double _expectedErf[];

    public:
        TEST_METHOD(gaussian_error_function___OK)
        {
            double* input       = _input;
            double* expectedErf = _expectedErf;

            for (int i = 0; i < 8; ++i)
            {
                double actual = gaussian_error_function(input[i]);

                Assert::AreEqual(expectedErf[i], actual, 1e-6);
            }
        }
        //---------------------------------------------------------------------
        TEST_METHOD(gaussian_error_function_complementary___OK)
        {
            double* input       = _input;
            double* expectedErf = _expectedErf;

            for (int i = 0; i < 8; ++i)
            {
                double actual = gaussian_error_function_complementary(input[i]);

                Assert::AreEqual(1.0 - expectedErf[i], actual, 1e-6);
            }
        }
        //---------------------------------------------------------------------
        TEST_METHOD(gaussian_error_function_vector___OK)
        {
            double* input       = _input;
            double* expectedErf = _expectedErf;
            double actual[8];

            gaussian_error_function_vector(input, actual, 8);

            for (int i = 0; i < 8; ++i)
                Assert::AreEqual(expectedErf[i], actual[i], 1e-6);
        }
        //---------------------------------------------------------------------
        TEST_METHOD(gaussian_error_function_complementary_vector___OK)
        {
            double* input       = _input;
            double* expectedErf = _expectedErf;
            double actual[8];

            gaussian_error_function_complementary_vector(input, actual, 8);

            for (int i = 0; i < 8; ++i)
                Assert::AreEqual(1.0 - expectedErf[i], actual[i], 1e-6);
        }
    };
    //-------------------------------------------------------------------------
    double special_functions_Tests::_input[]       = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    double special_functions_Tests::_expectedErf[] = {0.0, 0.5204999, 0.8427008, 0.9661051, 0.9953223, 0.9995930, 0.9999779, 0.9999993};
}
