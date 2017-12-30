using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private double CalculateSkewness()
        {
            return this.Count < ThresholdForParallel
                ? this.CalculateSkewnessSimd()
                : this.CalculateSkewnessParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double CalculateSkewnessSimd()
        {
            double skewness = this.CalculateSkewnessImpl((0, _values.Length));
            double sigma    = this.StandardDeviation;
            skewness /= _values.Length * sigma * sigma * sigma;

            return skewness;
        }
        //---------------------------------------------------------------------
        internal double CalculateSkewnessParallelizedSimd()
        {
            double skewness = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localSkewness = this.CalculateSkewnessImpl((range.Item1, range.Item2));
                    localSkewness.SafeAdd(ref skewness);
                }
            );

            double sigma = this.StandardDeviation;
            skewness /= _values.Length * sigma * sigma * sigma;

            return skewness;
        }
        //---------------------------------------------------------------------
        private unsafe double CalculateSkewnessImpl((int start, int end) range)
        {
            double skewness = 0;
            double avg      = this.Mean;
            var (i, n)      = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
                {
                    var avgVec  = new Vector<double>(avg);
                    var skewVec = new Vector<double>(0);

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        vec     -= avgVec;
                        skewVec += vec * vec * vec;

                        vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        vec     -= avgVec;
                        skewVec += vec * vec * vec;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        skewness += skewVec[j];
                }

                for (; i < n; ++i)
                {
                    double t = pArray[i] - avg;
                    skewness += t * t * t;
                }
            }

            return skewness;
        }
    }
}