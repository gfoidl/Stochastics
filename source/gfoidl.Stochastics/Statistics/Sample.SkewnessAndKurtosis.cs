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

            if (this.Count < ThresholdForParallel)
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
        private unsafe void CalculateSkewnessAndKurtosisImpl(int idxStart, int idxEnd, out double skewness, out double kurtosis)
        {
            double tmpSkewness = 0;
            double tmpKurtosis = 0;
            double avg         = this.Mean;

            fixed (double* pArray = _values)
            {
                double* start         = pArray + idxStart;
                double* end           = pArray + idxEnd;
                double* current       = start;
                double* sequentialEnd = default;
                int i                 = 0;
                int n                 = idxEnd - idxStart;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                    sequentialEnd = VectorHelper.GetAlignedPointer(start);
                else
                    sequentialEnd = end;

                // When SIMD is available, first pass is for register alignment.
                // Second pass will be the remaining elements.
            Sequential:
                while (current < sequentialEnd)
                {
                    double t     = *current - avg;
                    double t1    = t * t * t;
                    tmpSkewness += t1;
                    tmpKurtosis += t1 * t;
                    current++;
                }

                if (Vector.IsHardwareAccelerated)
                {
                    if (current >= end) goto Exit;

                    n -= (int)(current - start);

                    if (n < Vector<double>.Count)
                    {
                        sequentialEnd = end;
                        goto Sequential;
                    }

                    var avgVec   = new Vector<double>(avg);
                    var skewVec0 = Vector<double>.Zero;
                    var skewVec1 = Vector<double>.Zero;
                    var skewVec2 = Vector<double>.Zero;
                    var skewVec3 = Vector<double>.Zero;

                    var kurtVec0 = Vector<double>.Zero;
                    var kurtVec1 = Vector<double>.Zero;
                    var kurtVec2 = Vector<double>.Zero;
                    var kurtVec3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(current, 1 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);
                        Core(current, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref kurtVec2, end);
                        Core(current, 3 * Vector<double>.Count, avgVec, ref skewVec3, ref kurtVec3, end);
                        Core(current, 4 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(current, 5 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);
                        Core(current, 6 * Vector<double>.Count, avgVec, ref skewVec2, ref kurtVec2, end);
                        Core(current, 7 * Vector<double>.Count, avgVec, ref skewVec3, ref kurtVec3, end);

                        current += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(current, 1 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);
                        Core(current, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref kurtVec2, end);
                        Core(current, 3 * Vector<double>.Count, avgVec, ref skewVec3, ref kurtVec3, end);

                        current += 4 * Vector<double>.Count;
                        i       += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(current, 1 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);

                        current += 2 * Vector<double>.Count;
                        i       += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);

                        current += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    skewVec0    += skewVec1 + skewVec2 + skewVec3;
                    tmpSkewness += skewVec0.ReduceSum();

                    kurtVec0    += kurtVec1 + kurtVec2 + kurtVec3;
                    tmpKurtosis += kurtVec0.ReduceSum();

                    if (current < end)
                    {
                        sequentialEnd = end;
                        goto Sequential;
                    }
                }

            Exit:
                skewness = tmpSkewness;
                kurtosis = tmpKurtosis;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec, ref Vector<double> kurtVec, double* end)
            {
#if DEBUG_ASSERT
                // arr is included -> -1
                Debug.Assert(arr + offset + Vector<double>.Count - 1 < end);
#endif
                // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                vec               -= avgVec;
                Vector<double> tmp = vec * vec * vec;
                skewVec           += tmp;
                kurtVec           += tmp * vec;
            }
        }
    }
}
