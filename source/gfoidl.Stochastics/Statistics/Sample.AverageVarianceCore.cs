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
        private unsafe void CalculateAverageAndVarianceCoreImpl(int i, int n, out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
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
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec3, ref var3, end);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec3, ref var3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec3, ref var3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    avgVec0 += avgVec1 + avgVec2 + avgVec3;
                    tmpAvg   = avgVec0.ReduceSum();

                    var0        += var1 + var2 + var3;
                    tmpVariance  = var0.ReduceSum();
                }

                while (arr < end)
                {
                    tmpAvg      += *arr;
                    tmpVariance += *arr * *arr;
                    arr++;
                }

                avg      = tmpAvg;
                variance = tmpVariance;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec, ref Vector<double> var, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                avgVec            += vec;
                var               += vec * vec;
            }
        }
    }
}
