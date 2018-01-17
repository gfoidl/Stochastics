using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private void CalculateAverageAndVarianceCore()
        {
            var (avg, variance) = this.Count < ThresholdForParallel
                ? this.CalculateAverageAndVarianceCoreSimd()
                : this.CalculateAverageAndVarianceCoreParallelizedSimd();

            avg      /= this.Count;
            variance -= this.Count * avg * avg;

            _mean         = avg;
            _varianceCore = variance;
        }
        //---------------------------------------------------------------------
        internal (double avg, double variance) CalculateAverageAndVarianceCoreSimd()
        {
            var (avg, variance) = this.CalculateAverageAndVarianceCoreImpl((0, this.Count));

            return (avg, variance);
        }
        //---------------------------------------------------------------------
        internal (double avg, double variance) CalculateAverageAndVarianceCoreParallelizedSimd()
        {
            double avg      = 0;
            double variance = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    var (localAvg, localVariance) = this.CalculateAverageAndVarianceCoreImpl((range.Item1, range.Item2));
                    localAvg.SafeAdd(ref avg);
                    localVariance.SafeAdd(ref variance);
                }
            );

            return (avg, variance);
        }
        //---------------------------------------------------------------------
        private unsafe (double avg, double variance) CalculateAverageAndVarianceCoreImpl((int start, int end) range)
        {
            double avg      = 0;
            double variance = 0;
            var (i, n)      = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec = new Vector<double>(avg);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec, ref variance);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref variance);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref variance);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        avg += avgVec[j];
                }

                for (; i < n; ++i)
                {
                    avg      += pArray[i];
                    variance += pArray[i] * pArray[i];
                }
            }

            return (avg, variance);
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec, ref double var)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec += vec;
                var    += Vector.Dot(vec, vec);
            }
        }
    }
}