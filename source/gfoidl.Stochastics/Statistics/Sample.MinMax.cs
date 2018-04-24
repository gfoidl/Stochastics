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
                if (this.Count < ThresholdForMinMax)
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
        private unsafe void GetMinMaxImpl(int i, int n, out double min, out double max)
        {
            double tmpMin = double.MaxValue;
            double tmpMax = double.MinValue;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var minVec0 = new Vector<double>(tmpMin);
                    var minVec1 = new Vector<double>(tmpMin);
                    var minVec2 = new Vector<double>(tmpMin);
                    var minVec3 = new Vector<double>(tmpMin);

                    var maxVec0 = new Vector<double>(tmpMax);
                    var maxVec1 = new Vector<double>(tmpMax);
                    var maxVec2 = new Vector<double>(tmpMax);
                    var maxVec3 = new Vector<double>(tmpMax);

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec3, ref maxVec3, end);
                        Core(arr, 4 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 5 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 6 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(arr, 7 * Vector<double>.Count, ref minVec3, ref maxVec3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec3, ref maxVec3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction
                    minVec0 = Vector.Min(minVec0, Vector.Min(minVec1, Vector.Min(minVec2, minVec3)));
                    maxVec0 = Vector.Max(maxVec0, Vector.Max(maxVec1, Vector.Max(maxVec2, maxVec3)));
                    VectorHelper.ReduceMinMax(minVec0, maxVec0, ref tmpMin, ref tmpMax);
                }

                while (arr < end)
                {
                    if (*arr < tmpMin) tmpMin = *arr;
                    if (*arr > tmpMax) tmpMax = *arr;
                    arr++;
                }

                min = tmpMin;
                max = tmpMax;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> minVec, ref Vector<double> maxVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                minVec             = Vector.Min(minVec, vec);
                maxVec             = Vector.Max(maxVec, vec);
            }
        }
    }
}
