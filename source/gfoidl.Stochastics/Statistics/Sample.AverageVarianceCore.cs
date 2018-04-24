using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

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
                        Core(current, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(current, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(current, 2 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(current, 3 * Vector<double>.Count, ref avgVec3, ref var3, end);
                        Core(current, 4 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(current, 5 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(current, 6 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(current, 7 * Vector<double>.Count, ref avgVec3, ref var3, end);

                        current += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(current, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(current, 2 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(current, 3 * Vector<double>.Count, ref avgVec3, ref var3, end);

                        current += 4 * Vector<double>.Count;
                        i       += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(current, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);

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
                    avgVec0 += avgVec1 + avgVec2 + avgVec3;
                    tmpAvg  += avgVec0.ReduceSum();

                    var0        += var1 + var2 + var3;
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
                // Vector can be read aligned instead of unaligned, because arr was aligned 
                // in the sequential pass.
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec            += vec;
                var               += vec * vec;
            }
        }
    }
}
