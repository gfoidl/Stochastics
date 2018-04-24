using System;
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
        private double CalculateDelta()
        {
            return this.Count < ThresholdForParallel
                ? this.CalculateDeltaSimd()
                : this.CalculateDeltaParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double CalculateDeltaSimd()
        {
            double delta = this.CalculateDeltaImpl(0, _values.Length);

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
                    double localDelta = this.CalculateDeltaImpl(range.Item1, range.Item2);
                    localDelta.SafeAdd(ref delta);
                }
            );

            return delta / _values.Length;
        }
        //---------------------------------------------------------------------
        private unsafe double CalculateDeltaImpl(int i, int n)
        {
            double delta = 0;
            double avg   = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec    = new Vector<double>(avg);
                    var deltaVec0 = Vector<double>.Zero;
                    var deltaVec1 = Vector<double>.Zero;
                    var deltaVec2 = Vector<double>.Zero;
                    var deltaVec3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec3, end);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref deltaVec3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    deltaVec0 += deltaVec1 + deltaVec2 + deltaVec3;
                    delta      = deltaVec0.ReduceSum();
                }

                while (arr < end)
                {
                    delta += Math.Abs(*arr - avg);
                    arr++;
                }
            }

            return delta;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                deltaVec          += Vector.Abs(vec - avgVec);
            }
        }
    }
}
