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
            var sync        = new object();

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localVariance = VarianceCoreImpl((range.Item1, range.Item2));

                    lock (sync) variance += localVariance;
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

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
                {
                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> v1 = VectorHelper.GetVector(arr);
                        Vector<double> v2 = VectorHelper.GetVector(arr);
                        variance += Vector.Dot(v1, v2);
                        arr      += Vector<double>.Count;

                        v1 = VectorHelper.GetVector(arr);
                        v2 = VectorHelper.GetVector(arr);
                        variance += Vector.Dot(v1, v2);
                        arr      += Vector<double>.Count;
                    }
                }

                for (; i < n; ++i)
                    variance += pArray[i] * pArray[i];
            }

            return variance;
        }
    }
}