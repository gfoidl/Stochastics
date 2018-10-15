#pragma once
//-----------------------------------------------------------------------------
#ifdef GFOIDL_STOCHASTICS_GPU_EXPORTS
    #define GPU_API __declspec(dllexport)
#else
    #define GPU_API __declspec(dllimport)
#endif
//-----------------------------------------------------------------------------
#ifdef __cplusplus
    #define BEGIN_EXTERN_C  extern "C" {
    #define END_EXTERN_C    }
#else
    #define BEGIN_EXTERN_C
    #define END_EXTERN_C
#endif
