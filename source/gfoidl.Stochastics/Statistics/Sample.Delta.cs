using System;
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
        private double CalculateDelta()
        {
            return this.Count < SampleThresholds.ThresholdForParallel
                ? this.CalculateDeltaSimd()
                : this.CalculateDeltaParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double CalculateDeltaSimd()
        {
            double delta = this.CalculateDeltaImpl(_offset, _offset + _length);

            return delta / this.Count;
        }
        //---------------------------------------------------------------------
        internal double CalculateDeltaParallelizedSimd()
        {
            double delta = 0;

            Parallel.ForEach(
                Partitioner.Create(_offset, _offset + _length),
                range =>
                {
                    double localDelta = this.CalculateDeltaImpl(range.Item1, range.Item2);
                    localDelta.SafeAdd(ref delta);
                }
            );

            return delta / this.Count;
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
                        this.CalculateDeltaImplCore4(current, 0 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, ref deltaVec2, ref deltaVec3, end);
                        this.CalculateDeltaImplCore4(current, 4 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, ref deltaVec2, ref deltaVec3, end);

                        current += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        this.CalculateDeltaImplCore4(current, 0 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, ref deltaVec2, ref deltaVec3, end);

                        current += 4 * Vector<double>.Count;
                        i       += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        this.CalculateDeltaImplCore2(current, 0 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, end);

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
                    deltaVec0 += deltaVec1;
                    deltaVec2 += deltaVec3;
                    deltaVec0 += deltaVec2;
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
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void CalculateDeltaImplCore4(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec0, ref Vector<double> deltaVec1, ref Vector<double> deltaVec2, ref Vector<double> deltaVec3, double* end)
        {
#if DEBUG_ASSERT
            // arr is included -> -1
            Debug.Assert(arr + offset + 4 * Vector<double>.Count - 1 < end);
#endif
            // Vector can be read aligned instead of unaligned, because arr was aligned in the sequential pass.
            Vector<double> vec0 = VectorHelper.GetVector(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVector(arr + offset + 1 * Vector<double>.Count);
            Vector<double> vec2 = VectorHelper.GetVector(arr + offset + 2 * Vector<double>.Count);
            Vector<double> vec3 = VectorHelper.GetVector(arr + offset + 3 * Vector<double>.Count);

            vec0 -= avgVec;
            vec1 -= avgVec;
            vec2 -= avgVec;
            vec3 -= avgVec;

            deltaVec0 += Vector.Abs(vec0);
            deltaVec1 += Vector.Abs(vec1);
            deltaVec2 += Vector.Abs(vec2);
            deltaVec3 += Vector.Abs(vec3);
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void CalculateDeltaImplCore2(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec0, ref Vector<double> deltaVec1, double* end)
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

            deltaVec0 += Vector.Abs(vec0);
            deltaVec1 += Vector.Abs(vec1);
        }
    }
}
