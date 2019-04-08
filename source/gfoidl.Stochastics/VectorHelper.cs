using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

#if NETCOREAPP
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
#if DEBUG_ASSERT
            Debug.Assert(((long)arr % Unsafe.SizeOf<Vector<double>>()) == 0);
#endif
            return Unsafe.Read<Vector<double>>(arr);
        }
        //---------------------------------------------------------------------
        public static Vector<double> GetVectorUnaligned(double* arr)
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
#if DEBUG_ASSERT
            Debug.Assert(((long)arr % Unsafe.SizeOf<Vector<double>>()) == 0);
#endif
            Unsafe.Write(arr, vector);
        }
        //---------------------------------------------------------------------
        public static void WriteVectorUnaligned(this Vector<double> vector, double* arr)
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
#if NETCOREAPP3_0
            if (Avx.IsSupported)
            {
                Vector256<double> a     = Unsafe.As<Vector<double>, Vector256<double>>(ref vector);
                Vector256<double> tmp   = Avx.HorizontalAdd(a, a);
                Vector128<double> hi128 = tmp.GetUpper();
                Vector128<double> lo128 = tmp.GetLower();
                Vector128<double> s     = Sse2.Add(lo128, hi128);

                return s.ToScalar();
            }
#elif NETCOREAPP2_1
            if (Avx.IsSupported && Sse2.IsSupported)
            {
                Vector256<double> a     = Unsafe.As<Vector<double>, Vector256<double>>(ref vector);
                Vector256<double> tmp   = Avx.HorizontalAdd(a, a);
                Vector128<double> hi128 = Avx.ExtractVector128(tmp, 1);
                Vector128<double> lo128 = Avx.GetLowerHalf(tmp);
                Vector128<double> s     = Sse2.Add(lo128, hi128);

                return Sse2.ConvertToDouble(s);
            }
#endif
            return Vector.Dot(Vector<double>.One, vector);
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void ReduceMinMax(Vector<double> minVec, Vector<double> maxVec, ref double min, ref double max)
        {
#if NETCOREAPP3_0
            if (Avx.IsSupported)
            {
                min = MinMaxCore(minVec, true);
                max = MinMaxCore(maxVec, false);
                return;
            }
#elif NETCOREAPP2_1
            if (Avx.IsSupported && Sse2.IsSupported)
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
#if NETCOREAPP
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double MinMaxCore(Vector<double> vector, bool doMin)
        {
            Vector256<double> vec256 = Unsafe.As<Vector<double>, Vector256<double>>(ref vector);
#if NETCOREAPP3_0
            Vector128<double> hi128 = vec256.GetUpper();
            Vector128<double> lo128 = vec256.GetLower();
#elif NETCOREAPP2_1
            Vector128<double> hi128  = Avx.ExtractVector128(vec256, 1);
            Vector128<double> lo128  = Avx.GetLowerHalf(vec256);
#else
#error Update TFM
#endif
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

#if NETCOREAPP3_0
            return lo128.ToScalar();
#elif NETCOREAPP2_1
            return Sse2.ConvertToDouble(lo128);
#else
#error Update TFM
#endif
        }
#endif
            //---------------------------------------------------------------------
            // https://github.com/gfoidl/Stochastics/issues/47
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double* GetAlignedPointer(double* ptr)
        {
            const int bytesPerElement = sizeof(double) / sizeof(byte);
            // JIT will treat these as constants
            int sizeOfVector          = Unsafe.SizeOf<Vector<double>>();
            int vectorElements        = Vector<double>.Count;

            long address          = (long)ptr;
            int unalignedBytes    = (int)(address & (sizeOfVector - 1));    // address % sizeOfVector
            int unalignedElements = unalignedBytes / bytesPerElement;
            int elementsToAlign   = (vectorElements - unalignedElements) & (vectorElements - 1);

            double* aligned = ptr + elementsToAlign;
#if DEBUG_ASSERT
            Debug.Assert(((long)aligned % sizeOfVector) == 0);
            Debug.Assert(aligned - ptr >= 0);
#endif
            return aligned;
        }
    }
}
