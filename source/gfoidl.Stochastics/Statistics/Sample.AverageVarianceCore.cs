using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

#if DEBUG_ASSERT
using System.Diagnostics;
#endif

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private void CalculateAverageAndVarianceCore()
        {
            double avg      = 0;
            double variance = 0;

            if (this.Count < ThresholdForParallel)
                this.CalculateAverageAndVarianceCoreSimd(out avg, out variance);
            else
                this.CalculateAverageAndVarianceCoreParallelizedSimd(out avg, out variance);

            avg      /= this.Count;
            variance -= this.Count * avg * avg;

            _mean         = avg;
            _varianceCore = variance;
        }
        //---------------------------------------------------------------------
        internal void CalculateAverageAndVarianceCoreSimd(out double avg, out double variance)
        {
            this.CalculateAverageAndVarianceCoreImpl(0, this.Count, out avg, out variance);
        }
        //---------------------------------------------------------------------
        internal void CalculateAverageAndVarianceCoreParallelizedSimd(out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    this.CalculateAverageAndVarianceCoreImpl(range.Item1, range.Item2, out double localAvg, out double localVariance);
                    localAvg.SafeAdd(ref tmpAvg);
                    localVariance.SafeAdd(ref tmpVariance);
                }
            );

            avg      = tmpAvg;
            variance = tmpVariance;
        }
        //---------------------------------------------------------------------
        internal unsafe void CalculateAverageAndVarianceCoreImpl(int idxStart, int idxEnd, out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            fixed (double* pArray = _values)
            {
                double* start         = pArray + idxStart;
                double* end           = pArray + idxEnd;
                double* current       = start;
                double* sequentialEnd = default;
                int i                 = 0;
                int n                 = idxEnd - idxStart;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                    sequentialEnd = VectorHelper.GetAlignedPointer(start);
                else
                    sequentialEnd = end;

                // When SIMD is available, first pass is for register alignment.
                // Second pass will be the remaining elements.
            Sequential:
                while (current < sequentialEnd)
                {
                    tmpAvg      += *current;
                    tmpVariance += *current * *current;
                    current++;
                }

                if (Vector.IsHardwareAccelerated)
                {
                    if (current >= end) goto Exit;

                    n -= (int)(current - start);

                    if (n < Vector<double>.Count)
                    {
                        sequentialEnd = end;
                        goto Sequential;
                    }

                    var avgVec0 = Vector<double>.Zero;
                    var avgVec1 = Vector<double>.Zero;
                    var avgVec2 = Vector<double>.Zero;
                    var avgVec3 = Vector<double>.Zero;

                    var var0 = Vector<double>.Zero;
                    var var1 = Vector<double>.Zero;
                    var var2 = Vector<double>.Zero;
                    var var3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        CalculateAverageAndVarianceCoreImplCore4(current, 0 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref avgVec2, ref avgVec3, ref var0, ref var1, ref var2, ref var3, end);
                        CalculateAverageAndVarianceCoreImplCore4(current, 4 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref avgVec2, ref avgVec3, ref var0, ref var1, ref var2, ref var3, end);

                        current += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        CalculateAverageAndVarianceCoreImplCore4(current, 0 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref avgVec2, ref avgVec3, ref var0, ref var1, ref var2, ref var3, end);

                        current += 4 * Vector<double>.Count;
                        i       += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        CalculateAverageAndVarianceCoreImplCore2(current, 0 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref var0, ref var1, end);

                        current += 2 * Vector<double>.Count;
                        i       += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);

                        current += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    avgVec0 += avgVec1;
                    avgVec2 += avgVec3;
                    avgVec0 += avgVec2;
                    tmpAvg  += avgVec0.ReduceSum();

                    var0        += var1;
                    var2        += var3;
                    var0        += var2;
                    tmpVariance += var0.ReduceSum();

                    if (current < end)
                    {
                        sequentialEnd = end;
                        goto Sequential;            // second pass for sequential
                    }
                }

            Exit:
                avg      = tmpAvg;
                variance = tmpVariance;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec, ref Vector<double> var, double* end)
            {
#if DEBUG_ASSERT
                // arr is included -> -1
                Debug.Assert(arr + offset + Vector<double>.Count - 1 < end);
#endif
                // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec            += vec;
                var               += vec * vec;
            }
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void CalculateAverageAndVarianceCoreImplCore4(double* arr, int offset, ref Vector<double> avgVec0, ref Vector<double> avgVec1, ref Vector<double> avgVec2, ref Vector<double> avgVec3, ref Vector<double> var0, ref Vector<double> var1, ref Vector<double> var2, ref Vector<double> var3, double* end)
        {
#if DEBUG_ASSERT
            // arr is included -> -1
            Debug.Assert(arr + offset + 4 * Vector<double>.Count - 1 < end);
#endif
            // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
            Vector<double> vec0 = VectorHelper.GetVector(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVector(arr + offset + 1 * Vector<double>.Count);
            Vector<double> vec2 = VectorHelper.GetVector(arr + offset + 2 * Vector<double>.Count);
            Vector<double> vec3 = VectorHelper.GetVector(arr + offset + 3 * Vector<double>.Count);

            avgVec0 += vec0;
            avgVec1 += vec1;
            avgVec2 += vec2;
            avgVec3 += vec3;

            var tmp0 = vec0 * vec0;
            var tmp1 = vec1 * vec1;
            var tmp2 = vec2 * vec2;
            var tmp3 = vec3 * vec3;

            var0 += tmp0;
            var1 += tmp1;
            var2 += tmp2;
            var3 += tmp3;
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void CalculateAverageAndVarianceCoreImplCore2(double* arr, int offset, ref Vector<double> avgVec0, ref Vector<double> avgVec1, ref Vector<double> var0, ref Vector<double> var1, double* end)
        {
#if DEBUG_ASSERT
            // arr is included -> -1
            Debug.Assert(arr + offset + 2 * Vector<double>.Count - 1 < end);
#endif
            // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
            Vector<double> vec0 = VectorHelper.GetVector(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVector(arr + offset + 1 * Vector<double>.Count);

            avgVec0 += vec0;
            avgVec1 += vec1;

            var tmp0 = vec0 * vec0;
            var tmp1 = vec1 * vec1;

            var0 += tmp0;
            var1 += tmp1;
        }
    }
}
