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
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReduceMinMax(Vector<double> minVec, Vector<double> maxVec, ref double min, ref double max)
        {
#if NETCOREAPP2_1
            if (Avx.IsSupported && Sse2.IsSupported && 256 / 8 == sizeof(double) * Vector<double>.Count)
            {
                min = MinMaxCore(minVec, true);
                max = MinMaxCore(maxVec, false);
                return;
            }
#endif
            for (int j = 0; j < Vector<double>.Count; ++j)
            {
                if (minVec[j] < min) min = minVec[j];
                if (maxVec[j] > max) max = maxVec[j];
            }
        }
        //---------------------------------------------------------------------
#if NETCOREAPP2_1
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double MinMaxCore(Vector<double> vector, bool doMin)
        {
            Vector256<double> vec256 = Unsafe.As<Vector<double>, Vector256<double>>(ref vector);
            Vector128<double> hi128  = Avx.ExtractVector128(vec256, 1);
            Vector128<double> lo128  = Avx.ExtractVector128(vec256, 0);
            Vector128<double> tmp1   = Avx.Permute(hi128, 0b_01);
            Vector128<double> tmp2   = Avx.Permute(lo128, 0b_01);

            if (doMin)
            {
                hi128 = Sse2.Min(hi128, tmp1);
                lo128 = Sse2.Min(lo128, tmp2);
                lo128 = Sse2.Min(lo128, hi128);
            }
            else
            {
                hi128 = Sse2.Max(hi128, tmp1);
                lo128 = Sse2.Max(lo128, tmp2);
                lo128 = Sse2.Max(lo128, hi128);
            }

            return Sse2.ConvertToDouble(lo128);
        }
#endif
    }
}
