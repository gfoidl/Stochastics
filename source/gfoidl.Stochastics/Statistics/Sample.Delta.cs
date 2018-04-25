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
        private unsafe double CalculateDeltaImpl(int idxStart, int idxEnd)
        {
            double delta = 0;
            double avg   = this.Mean;

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
                    delta += Math.Abs(*current - avg);
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

                    var avgVec    = new Vector<double>(avg);
                    var deltaVec0 = Vector<double>.Zero;
                    var deltaVec1 = Vector<double>.Zero;
                    var deltaVec2 = Vector<double>.Zero;
                    var deltaVec3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(current, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(current, 2 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(current, 3 * Vector<double>.Count, avgVec, ref deltaVec3, end);
                        Core(current, 4 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(current, 5 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(current, 6 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(current, 7 * Vector<double>.Count, avgVec, ref deltaVec3, end);

                        current += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(current, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(current, 2 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(current, 3 * Vector<double>.Count, avgVec, ref deltaVec3, end);

                        current += 4 * Vector<double>.Count;
                        i       += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(current, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);

                        current += 2 * Vector<double>.Count;
                        i       += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(current, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);

                        current += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    deltaVec0 += deltaVec1 + deltaVec2 + deltaVec3;
                    delta     += deltaVec0.ReduceSum();

                    if (current < end)
                    {
                        sequentialEnd = end;
                        goto Sequential;            // second pass for sequential
                    }
                }

            Exit:
                return delta;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec, double* end)
            {
#if DEBUG_ASSERT
                // arr is included -> -1
                Debug.Assert(arr + offset + Vector<double>.Count - 1 < end);
#endif
                // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                deltaVec          += Vector.Abs(vec - avgVec);
            }
        }
    }
}
