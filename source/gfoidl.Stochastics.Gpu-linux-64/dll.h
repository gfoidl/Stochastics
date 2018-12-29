#pragma once
//-----------------------------------------------------------------------------
#if defined _WIN32 || defined __CYGWIN__
    #ifdef GFOIDL_STOCHASTICS_GPU_EXPORTS
        #define GPU_API __declspec(dllexport)
    #else
        #define GPU_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define GPU_API __attribute__ ((visibility ("default")))
    #else
        #define GPU_API
    #endif
#endif
//-----------------------------------------------------------------------------
#ifdef __cplusplus
    #define BEGIN_EXTERN_C  extern "C" {
    #define END_EXTERN_C    }
#else
    #define BEGIN_EXTERN_C
    #define END_EXTERN_C
#endif
