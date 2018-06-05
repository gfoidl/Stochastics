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
        private void GetMinMax()
        {
            if (this.Count == 1)
            {
                _min = _values[0];
                _max = _values[0];
            }
            else
            {
                if (this.Count < SampleThresholds.ThresholdForMinMax)
                    this.GetMinMaxSimd(out _min, out _max);
                else
                    this.GetMinMaxParallelizedSimd(out _min, out _max);
            }
        }
        //---------------------------------------------------------------------
        internal void GetMinMaxSimd(out double min, out double max)
        {
            this.GetMinMaxImpl(0, this.Count, out min, out max);
        }
        //---------------------------------------------------------------------
        internal void GetMinMaxParallelizedSimd(out double min, out double max)
        {
            double tmpMin = double.MaxValue;
            double tmpMax = double.MinValue;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    this.GetMinMaxImpl(range.Item1, range.Item2, out double localMin, out double localMax);
                    localMin.InterlockedExchangeIfSmaller(ref tmpMin);
                    localMax.InterlockedExchangeIfGreater(ref tmpMax);
                }
            );

            min = tmpMin;
            max = tmpMax;
        }
        //---------------------------------------------------------------------
        private unsafe void GetMinMaxImpl(int idxStart, int idxEnd, out double min, out double max)
        {
            double tmpMin = double.MaxValue;
            double tmpMax = double.MinValue;

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
                    if (*current < tmpMin) tmpMin = *current;
                    if (*current > tmpMax) tmpMax = *current;
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

                    var minVec0 = new Vector<double>(tmpMin);
                    var minVec1 = minVec0;
                    var minVec2 = minVec0;
                    var minVec3 = minVec0;

                    var maxVec0 = new Vector<double>(tmpMax);
                    var maxVec1 = maxVec0;
                    var maxVec2 = maxVec0;
                    var maxVec3 = maxVec0;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(current, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(current, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(current, 2 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(current, 3 * Vector<double>.Count, ref minVec3, ref maxVec3, end);
                        Core(current, 4 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(current, 5 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(current, 6 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(current, 7 * Vector<double>.Count, ref minVec3, ref maxVec3, end);

                        current += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(current, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(current, 2 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(current, 3 * Vector<double>.Count, ref minVec3, ref maxVec3, end);

                        current += 4 * Vector<double>.Count;
                        i       += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(current, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        current += 2 * Vector<double>.Count;
                        i       += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);

                        current += 1 * Vector<double>.Count;
                    }

                    // Reduction
                    minVec0 = Vector.Min(minVec0, minVec1);
                    minVec2 = Vector.Min(minVec2, minVec3);
                    maxVec0 = Vector.Max(maxVec0, maxVec1);
                    maxVec2 = Vector.Max(maxVec2, maxVec3);

                    minVec0 = Vector.Min(minVec0, minVec2);
                    maxVec0 = Vector.Max(maxVec0, maxVec2);
                    VectorHelper.ReduceMinMax(minVec0, maxVec0, ref tmpMin, ref tmpMax);

                    if (current < end)
                    {
                        sequentialEnd = end;
                        goto Sequential;
                    }
                }

            Exit:
                min = tmpMin;
                max = tmpMax;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> minVec, ref Vector<double> maxVec, double* end)
            {
#if DEBUG_ASSERT
                // arr is included -> -1
                Debug.Assert(arr + offset + Vector<double>.Count - 1 < end);
#endif
                // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                minVec             = Vector.Min(minVec, vec);
                maxVec             = Vector.Max(maxVec, vec);
            }
        }
    }
}
