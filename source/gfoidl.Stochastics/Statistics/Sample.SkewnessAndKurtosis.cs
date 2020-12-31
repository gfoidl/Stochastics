using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

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

            if (this.Count < SampleThresholds.ThresholdForParallel)
                this.CalculateSkewnessAndKurtosisSimd(out skewness, out kurtosis);
            else
                this.CalculateSkewnessAndKurtosisParallelizedSimd(out skewness, out kurtosis);

            double sigma = this.StandardDeviation;
            double t     = this.Count * sigma * sigma * sigma;
            skewness    /= t;
            kurtosis    /= t * sigma;

            _skewness = skewness;
            _kurtosis = kurtosis;
        }
        //---------------------------------------------------------------------
        internal void CalculateSkewnessAndKurtosisSimd(out double skewness, out double kurtosis)
        {
            this.CalculateSkewnessAndKurtosisImpl(_offset, _offset + _length, out skewness, out kurtosis);
        }
        //---------------------------------------------------------------------
        internal void CalculateSkewnessAndKurtosisParallelizedSimd(out double skewness, out double kurtosis)
        {
            double tmpSkewness = 0;
            double tmpKurtosis = 0;

            Parallel.ForEach(
                Partitioner.Create(_offset, _offset + _length),
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
                        // A ...Core4 doesn't work, because of register spilling. Here up to ymm13 is used.
                        CalculateSkewnessAndKurtosisImplCore2(current, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);
                        CalculateSkewnessAndKurtosisImplCore2(current, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref skewVec3, ref kurtVec2, ref kurtVec3, end);
                        CalculateSkewnessAndKurtosisImplCore2(current, 4 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);
                        CalculateSkewnessAndKurtosisImplCore2(current, 6 * Vector<double>.Count, avgVec, ref skewVec2, ref skewVec3, ref kurtVec2, ref kurtVec3, end);

                        current += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        CalculateSkewnessAndKurtosisImplCore2(current, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);
                        CalculateSkewnessAndKurtosisImplCore2(current, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref skewVec3, ref kurtVec2, ref kurtVec3, end);

                        current += 4 * Vector<double>.Count;
                        i       += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        CalculateSkewnessAndKurtosisImplCore2(current, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);

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
                    skewVec0    += skewVec1;
                    skewVec2    += skewVec3;
                    skewVec0    += skewVec2;
                    tmpSkewness += skewVec0.ReduceSum();

                    kurtVec0    += kurtVec1;
                    kurtVec2    += kurtVec3;
                    kurtVec0    += kurtVec2;
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
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void CalculateSkewnessAndKurtosisImplCore2(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec0, ref Vector<double> skewVec1, ref Vector<double> kurtVec0, ref Vector<double> kurtVec1, double* end)
        {
#if DEBUG_ASSERT
            // arr is included -> -1
            Debug.Assert(arr + offset + 2 * Vector<double>.Count - 1 < end);
#endif
            // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
            Vector<double> vec0 = VectorHelper.GetVector(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVector(arr + offset + 1 * Vector<double>.Count);

            vec0 -= avgVec;
            vec1 -= avgVec;

            var tmp0 = vec0 * vec0;
            var tmp1 = vec1 * vec1;

            tmp0 *= vec0;
            tmp1 *= vec1;

            skewVec0 += tmp0;
            skewVec1 += tmp1;

            tmp0 *= vec0;
            tmp1 *= vec1;

            kurtVec0 += tmp0;
            kurtVec1 += tmp1;
        }
    }
}
