using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

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

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    tmpSkewness += skewVec.ReduceSum();
                    tmpKurtosis += kurtVec.ReduceSum();
                }

                for (; i < n; ++i)
                {
                    double t  = pArray[i] - avg;
                    double t1 = t * t * t;
                    tmpSkewness += t1;
                    tmpKurtosis += t1 * t;
                }

                skewness = tmpSkewness;
                kurtosis = tmpKurtosis;
            }
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
