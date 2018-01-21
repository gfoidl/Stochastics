using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private double CalculateDelta()
        {
            this.EnsureValuesInitialized();

            return this.Count < ThresholdForParallel
                ? this.CalculateDeltaSimd()
                : this.CalculateDeltaParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double CalculateDeltaSimd()
        {
            double delta = this.CalculateDeltaImpl((0, _values.Length));

            return delta / _values.Length;
        }
        //---------------------------------------------------------------------
        internal double CalculateDeltaParallelizedSimd()
        {
            double delta = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localDelta = this.CalculateDeltaImpl((range.Item1, range.Item2));
                    localDelta.SafeAdd(ref delta);
                }
            );

            return delta / _values.Length;
        }
        //---------------------------------------------------------------------
        private unsafe double CalculateDeltaImpl((int start, int end) range)
        {
            double delta = 0;
            double avg   = this.Mean;
            var (i, n)   = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec   = new Vector<double>(avg);
                    var deltaVec = new Vector<double>(0);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref deltaVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        delta += deltaVec[j];
                }

                for (; i < n; ++i)
                    delta += Math.Abs(pArray[i] - avg);
            }

            return delta;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                deltaVec += Vector.Abs(vec - avgVec);
            }
        }
    }
}