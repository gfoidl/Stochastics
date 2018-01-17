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

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localKurtosis = this.CalculateKurtosisImpl((range.Item1, range.Item2));
                    localKurtosis.SafeAdd(ref kurtosis);
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

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec  = new Vector<double>(avg);
                    var kurtVec = new Vector<double>(0);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref kurtVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref kurtVec);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref kurtVec);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec);

                        i += Vector<double>.Count;
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
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> kurtVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                vec     -= avgVec;
                kurtVec += vec * vec * vec * vec;
            }
        }
    }
}