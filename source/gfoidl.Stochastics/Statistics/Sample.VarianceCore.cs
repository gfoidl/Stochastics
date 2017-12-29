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
            double variance = 0;
            int n           = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
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

            double avg = this.Mean;
            variance -= n * avg * avg;

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
                    double local = 0;
                    int n        = range.Item2;

                    fixed (double* pArray = _values)
                    {
                        int i       = range.Item1;
                        double* arr = pArray + i;

                        if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
                        {
                            for (; i < range.Item2 - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                            {
                                Vector<double> v1 = VectorHelper.GetVector(arr);
                                Vector<double> v2 = VectorHelper.GetVector(arr);
                                local += Vector.Dot(v1, v2);
                                arr   += Vector<double>.Count;

                                v1 = VectorHelper.GetVector(arr);
                                v2 = VectorHelper.GetVector(arr);
                                local += Vector.Dot(v1, v2);
                                arr   += Vector<double>.Count;
                            }
                        }

                        for (; i < range.Item2; ++i)
                            local += pArray[i] * pArray[i];
                    }

                    lock (sync) variance += local;
                }
            );

            double avg = this.Mean;
            variance -= this.Count * avg * avg;

            return variance;
        }
    }
}