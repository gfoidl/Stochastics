using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private double CalculateVarianceCore()
        {
            return this.Count < ThresholdForParallel
                ? this.VarianceCoreSimd()
                : this.VarianceCoreParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal unsafe double VarianceCoreSimd()
        {
            double variance = this.VarianceCoreImpl((0, _values.Length));
            double avg      = this.Mean;
            variance -= _values.Length * avg * avg;

            return variance;
        }
        //---------------------------------------------------------------------
        internal unsafe double VarianceCoreParallelizedSimd()
        {
            double variance = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localVariance = this.VarianceCoreImpl((range.Item1, range.Item2));
                    localVariance.SafeAdd(ref variance);
                }
            );

            double avg = this.Mean;
            variance -= this.Count * avg * avg;

            return variance;
        }
        //---------------------------------------------------------------------
        private unsafe double VarianceCoreImpl((int start, int end) range)
        {
            double variance = 0;
            var (i, n)      = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref variance);
                        Core(arr, 2 * Vector<double>.Count, ref variance);
                        Core(arr, 3 * Vector<double>.Count, ref variance);
                        Core(arr, 4 * Vector<double>.Count, ref variance);
                        Core(arr, 5 * Vector<double>.Count, ref variance);
                        Core(arr, 6 * Vector<double>.Count, ref variance);
                        Core(arr, 7 * Vector<double>.Count, ref variance);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref variance);
                        Core(arr, 2 * Vector<double>.Count, ref variance);
                        Core(arr, 3 * Vector<double>.Count, ref variance);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref variance);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref variance);

                        i += Vector<double>.Count;
                    }
                }

                for (; i < n; ++i)
                    variance += pArray[i] * pArray[i];
            }

            return variance;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref double var)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                var += Vector.Dot(vec, vec);
            }
        }
    }
}