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
        private void CalculateSkewnessAndKurtosis()
        {
            double skewness = 0;
            double kurtosis = 0;

            if (this.Count < ThresholdForSkewnessAndKurtosis)
                this.CalculateSkewnessAndKurtosisSimd(out skewness, out kurtosis);
            else
                this.CalculateSkewnessAndKurtosisParallelizedSimd(out skewness, out kurtosis);

            double sigma = this.StandardDeviation;
            double t     = _values.Length * sigma * sigma * sigma;
            skewness    /= t;
            kurtosis    /= t * sigma;

            _skewness = skewness;
            _kurtosis = kurtosis;
        }
        //---------------------------------------------------------------------
        internal void CalculateSkewnessAndKurtosisSimd(out double skewness, out double kurtosis)
        {
            this.CalculateSkewnessAndKurtosisImpl(0, this.Count, out skewness, out kurtosis);
        }
        //---------------------------------------------------------------------
        internal void CalculateSkewnessAndKurtosisParallelizedSimd(out double skewness, out double kurtosis)
        {
            double tmpSkewness = 0;
            double tmpKurtosis = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    this.CalculateSkewnessAndKurtosisImpl(range.Item1, range.Item2, out double localSkewness, out double localKurtosis);
                    localSkewness.SafeAdd(ref tmpSkewness);
                    localKurtosis.SafeAdd(ref tmpKurtosis);
                }
            );

            skewness = tmpSkewness;
            kurtosis = tmpKurtosis;
        }
        //---------------------------------------------------------------------
        private unsafe void CalculateSkewnessAndKurtosisImpl(int i, int n, out double skewness, out double kurtosis)
        {
            double tmpSkewness   = 0;
            double tmpKurtosis   = 0;
            double avg           = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec  = new Vector<double>(avg);
                    var skewVec = Vector<double>.Zero;
                    var kurtVec = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    tmpSkewness += skewVec.ReduceSum();
                    tmpKurtosis += kurtVec.ReduceSum();
                }

                while (arr < end)
                {
                    double t     = *arr - avg;
                    double t1    = t * t * t;
                    tmpSkewness += t1;
                    tmpKurtosis += t1 * t;
                    arr++;
                }

                skewness = tmpSkewness;
                kurtosis = tmpKurtosis;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec, ref Vector<double> kurtVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                vec               -= avgVec;
                Vector<double> tmp = vec * vec * vec;
                skewVec           += tmp;
                kurtVec           += tmp * vec;
            }
        }
    }
}
