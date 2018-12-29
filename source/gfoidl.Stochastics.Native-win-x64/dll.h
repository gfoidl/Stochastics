#pragma once
//-----------------------------------------------------------------------------
#ifdef GFOIDL_STOCHASTICS_NATIVE_EXPORTS
    #define DLL_API __declspec(dllexport)
#else
    #define DLL_API __declspec(dllimport)
#endif
