using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

#if NETCOREAPP2_1
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace gfoidl.Stochastics
{
    [DebuggerNonUserCode]
    internal static unsafe class VectorHelper
    {
        public static Vector<double> GetVector(double* arr)
        {
            return Unsafe.ReadUnaligned<Vector<double>>(arr);
        }
        //---------------------------------------------------------------------
        [Obsolete("Use GetVector + offset instead")]
        public static Vector<double> GetVectorWithAdvance(ref double* arr)
        {
            Vector<double> vec = Unsafe.ReadUnaligned<Vector<double>>(arr);
            arr += Vector<double>.Count;

            return vec;
        }
        //---------------------------------------------------------------------
        public static void WriteVector(this Vector<double> vector, double* arr)
        {
            Unsafe.WriteUnaligned(arr, vector);
        }
        //---------------------------------------------------------------------
        [Obsolete("Use WriteVector + offset instead")]
        public static void WriteVectorWithAdvance(this Vector<double> vector, ref double* arr)
        {
            Unsafe.WriteUnaligned(arr, vector);
            arr += Vector<double>.Count;
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ReduceSum(this Vector<double> vector)
        {
#if NETCOREAPP2_1
            if (Avx.IsSupported && Sse2.IsSupported && 256 / 8 == sizeof(double) * Vector<double>.Count)
            {
                Vector256<double> a     = Unsafe.As<Vector<double>, Vector256<double>>(ref vector);
                Vector256<double> tmp   = Avx.HorizontalAdd(a, a);
                Vector128<double> hi128 = Avx.ExtractVector128(tmp, 1);
                Vector128<double> s     = Sse2.Add(Unsafe.As<Vector256<double>, Vector128<double>>(ref tmp), hi128);

                return Sse2.ConvertToDouble(s);
            }
#endif
            return Vector.Dot(Vector<double>.One, vector);
        }
    }
}
