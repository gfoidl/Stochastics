using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private double CalculateAverage()
        {
            return this.Count < ThresholdForParallel
                ? this.CalculateAverageSimd()
                : this.CalculateAverageParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double CalculateAverageSimd()
        {
            double avg = this.CalculateAverageImpl((0, this.Count));

            return avg / this.Count;
        }
        //---------------------------------------------------------------------
        internal double CalculateAverageParallelizedSimd()
        {
            double avg = 0;

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);

            Parallel.ForEach(
                partitioner,
                range =>
                {
                    double tmp = this.CalculateAverageImpl((range.Item1, range.Item2));
                    tmp.SafeAdd(ref avg);
                }
            );

            return avg / this.Count;
        }
        //---------------------------------------------------------------------
        private unsafe double CalculateAverageImpl((int start, int end) range)
        {
            double avg = 0;
            var (i, n) = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec = new Vector<double>(avg);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        avg += avgVec[j];
                }

                for (; i < n; ++i)
                    avg += pArray[i];
            }

            return avg;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec += vec;
            }
        }
    }
}