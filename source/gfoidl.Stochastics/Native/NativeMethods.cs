using System.Runtime.InteropServices;
using System.Security;

namespace gfoidl.Stochastics.Native
{
    internal static class NativeMethods
    {
        private const string LibName = "gfoidl-Stochastics-Native";
        //---------------------------------------------------------------------
        [DllImport(LibName)]
        [SuppressUnmanagedCodeSecurity]
        public static extern double gaussian_error_function(double x);
        //---------------------------------------------------------------------
        [DllImport(LibName)]
        [SuppressUnmanagedCodeSecurity]
        public static extern double gaussian_error_function_complementary(double x);
        //---------------------------------------------------------------------
        [DllImport(LibName)]
        [SuppressUnmanagedCodeSecurity]
        public static extern unsafe void gaussian_error_function_vector(double* values, double* result, int n);
        //---------------------------------------------------------------------
        [DllImport(LibName)]
        [SuppressUnmanagedCodeSecurity]
        public static extern unsafe void gaussian_error_function_complementary_vector(double* values, double* result, int n);
    }
}
