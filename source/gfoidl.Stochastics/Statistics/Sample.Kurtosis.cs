using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private double CalculateKurtosis()
        {
            return this.Count < ThresholdForParallel
                ? this.CalculateKurtosisSimd()
                : this.CalculateKurtosisParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal unsafe double CalculateKurtosisSimd()
        {
            double kurtosis = 0;
            double avg      = this.Mean;
            int n           = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    var avgVec  = new Vector<double>(avg);
                    var kurtVec = new Vector<double>(0);

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        vec     -= avgVec;
                        kurtVec += vec * vec * vec * vec;

                        vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        vec     -= avgVec;
                        kurtVec += vec * vec * vec * vec;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        kurtosis += kurtVec[j];
                }

                for (; i < n; ++i)
                {
                    double t = pArray[i] - avg;
                    kurtosis += t * t * t * t;
                }
            }

            double sigma = this.StandardDeviation;
            kurtosis /= n * sigma * sigma * sigma * sigma;

            return kurtosis;
        }
        //---------------------------------------------------------------------
        internal unsafe double CalculateKurtosisParallelizedSimd()
        {
            double kurtosis = 0;
            var sync        = new object();

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double local = 0;
                    double avg   = this.Mean;
                    int n        = range.Item2;

                    fixed (double* pArray = _values)
                    {
                        int i       = range.Item1;
                        double* arr = pArray + i;

                        if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
                        {
                            var avgVec  = new Vector<double>(avg);
                            var kurtVec = new Vector<double>(0);

                            for (; i < range.Item2 - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                            {
                                Vector<double> vec = VectorHelper.GetVectorWithAdvance(ref arr);
                                vec     -= avgVec;
                                kurtVec += vec * vec * vec * vec;

                                vec = VectorHelper.GetVectorWithAdvance(ref arr);
                                vec     -= avgVec;
                                kurtVec += vec * vec * vec * vec;
                            }

                            for (int j = 0; j < Vector<double>.Count; ++j)
                                local += kurtVec[j];
                        }

                        for (; i < n; ++i)
                        {
                            double t = pArray[i] - avg;
                            local += t * t * t * t;
                        }
                    }

                    lock (sync) kurtosis += local;
                }
            );

            double sigma = this.StandardDeviation;
            kurtosis /= _values.Length * sigma * sigma * sigma * sigma;

            return kurtosis;
        }
    }
}