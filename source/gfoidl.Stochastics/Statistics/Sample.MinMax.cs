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

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var minVec = new Vector<double>(tmpMin);
                    var maxVec = new Vector<double>(tmpMax);

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

                    // Reduction
                    VectorHelper.ReduceMinMax(minVec, maxVec, ref tmpMin, ref tmpMax);
                }

                for (; i < n; ++i)
                {
                    if (pArray[i] < tmpMin) tmpMin = pArray[i];
                    if (pArray[i] > tmpMax) tmpMax = pArray[i];
                }

                min = tmpMin;
                max = tmpMax;
            }
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
