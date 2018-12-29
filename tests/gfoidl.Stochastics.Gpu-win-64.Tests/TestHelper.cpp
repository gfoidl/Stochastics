#include "stdafx.h"
#include "TestHelper.h"
#include "gpu_core.h"
//-----------------------------------------------------------------------------
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
//-----------------------------------------------------------------------------
void TestHelper::FailIfError(const int errorCode)
{
    if (errorCode == 0) return;

    const char* errorMsg = gpu_get_error_string(errorCode);
    wchar_t* msg         = TestHelper::ToWchar(errorMsg);
    Assert::Fail(msg);
}
//-----------------------------------------------------------------------------
wchar_t* TestHelper::ToWchar(const char* c)
{
    const size_t strLen = strlen(c) + 1;
    wchar_t* wc         = new wchar_t[strLen];
    mbstowcs(wc, c, strLen);

    return wc;
}
