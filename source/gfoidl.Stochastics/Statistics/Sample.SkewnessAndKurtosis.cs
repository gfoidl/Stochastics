using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private void CalculateSkewnessAndKurtosis()
        {
            this.EnsureValuesInitialized();

            var (skewness, kurtosis) = this.Count < ThresholdForSkewnessAndKurtosis
                ? this.CalculateSkewnessAndKurtosisSimd()
                : this.CalculateSkewnessAndKurtosisParallelizedSimd();

            double sigma = this.StandardDeviation;
            double t     = _values.Length * sigma * sigma * sigma;
            skewness /= t;
            kurtosis /= t * sigma;

            _skewness = skewness;
            _kurtosis = kurtosis;
        }
        //---------------------------------------------------------------------
        internal (double skewness, double kurtosis) CalculateSkewnessAndKurtosisSimd()
        {
            var (skewness, kurtosis) = this.CalculateSkewnessAndKurtosisImpl((0, this.Count));

            return (skewness, kurtosis);
        }
        //---------------------------------------------------------------------
        internal (double skewness, double kurtosis) CalculateSkewnessAndKurtosisParallelizedSimd()
        {
            double skewness = 0;
            double kurtosis = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    var (localSkewness, localKurtosis) = this.CalculateSkewnessAndKurtosisImpl((range.Item1, range.Item2));
                    localSkewness.SafeAdd(ref skewness);
                    localKurtosis.SafeAdd(ref kurtosis);
                }
            );

            return (skewness, kurtosis);
        }
        //---------------------------------------------------------------------
        private unsafe (double skewness, double kurtosis) CalculateSkewnessAndKurtosisImpl((int start, int end) range)
        {
            double skewness = 0;
            double kurtosis = 0;
            double avg      = this.Mean;
            var (i, n)      = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec  = new Vector<double>(avg);
                    var skewVec = new Vector<double>(0);
                    var kurtVec = new Vector<double>(0);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                    {
                        skewness += skewVec[j];
                        kurtosis += kurtVec[j];
                    }
                }

                for (; i < n; ++i)
                {
                    double t  = pArray[i] - avg;
                    double t1 = t * t * t;
                    skewness += t1;
                    kurtosis += t1 * t;
                }
            }

            return (skewness, kurtosis);
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec, ref Vector<double> kurtVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                vec -= avgVec;
                Vector<double> tmp = vec * vec * vec;
                skewVec += tmp;
                kurtVec += tmp * vec;
            }
        }
    }
}