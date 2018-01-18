using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

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
                var (min, max) = this.Count < ThresholdForMinMax
                    ? this.GetMinMaxSimd()
                    : this.GetMinMaxParallelizedSimd();

                _min = min;
                _max = max;
            }
        }
        //---------------------------------------------------------------------
        internal (double min, double max) GetMinMaxSimd()
        {
            var (min, max) = this.GetMinMaxImpl((0, this.Count));

            return (min, max);
        }
        //---------------------------------------------------------------------
        internal (double min, double max) GetMinMaxParallelizedSimd()
        {
            double min = double.MaxValue;
            double max = double.MinValue;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    var (localMin, localMax) = this.GetMinMaxImpl((range.Item1, range.Item2));
                    localMin.InterlockedExchangeIfSmaller(ref min);
                    localMax.InterlockedExchangeIfGreater(ref max);
                }
            );

            return (min, max);
        }
        //---------------------------------------------------------------------
        private unsafe (double min, double max) GetMinMaxImpl((int start, int end) range)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            var (i, n) = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var minVec = new Vector<double>(min);
                    var maxVec = new Vector<double>(max);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 2 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 3 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 4 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 5 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 6 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 7 * Vector<double>.Count, ref minVec, ref maxVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 2 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 3 * Vector<double>.Count, ref minVec, ref maxVec);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                    {
                        if (minVec[j] < min) min = minVec[j];
                        if (maxVec[j] > max) max = maxVec[j];
                    }
                }

                for (; i < n; ++i)
                {
                    if (pArray[i] < min) min = pArray[i];
                    if (pArray[i] > max) max = pArray[i];
                }
            }

            return (min, max);
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> minVec, ref Vector<double> maxVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                minVec             = Vector.Min(minVec, vec);
                maxVec             = Vector.Max(maxVec, vec);
            }
        }
    }
}