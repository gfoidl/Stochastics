#pragma once
//-----------------------------------------------------------------------------
class TestHelper
{
public:
    static void FailIfError(const int errorCode);

private:
    static wchar_t* ToWchar(const char* c);
};
