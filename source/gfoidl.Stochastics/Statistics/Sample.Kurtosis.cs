using System.Collections.Concurrent;
using System.Numerics;
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
        internal double CalculateKurtosisSimd()
        {
            double kurtosis = this.CalculateKurtosisImpl((0, _values.Length));
            double sigma    = this.StandardDeviation;
            kurtosis /= _values.Length * sigma * sigma * sigma * sigma;

            return kurtosis;
        }
        //---------------------------------------------------------------------
        internal double CalculateKurtosisParallelizedSimd()
        {
            double kurtosis = 0;
            var sync        = new object();

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localKurtosis = this.CalculateKurtosisImpl((range.Item1, range.Item2));

                    lock (sync) kurtosis += localKurtosis;
                }
            );

            double sigma = this.StandardDeviation;
            kurtosis /= _values.Length * sigma * sigma * sigma * sigma;

            return kurtosis;
        }
        //---------------------------------------------------------------------
        private unsafe double CalculateKurtosisImpl((int start, int end) range)
        {
            double kurtosis = 0;
            double avg      = this.Mean;
            var (i, n)      = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
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

            return kurtosis;
        }
    }
}