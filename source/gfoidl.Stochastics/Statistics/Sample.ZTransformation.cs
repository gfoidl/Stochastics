using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

#if DEBUG_ASSERT
using System.Diagnostics;
#endif

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        internal IEnumerable<(double Value, double zTransformed)> ZTransformationInternal(double? standardDeviation = null)
        {
            double avg   = this.Mean;
            double sigma = standardDeviation ?? this.SampleStandardDeviation;
            double[] tmp = _values;

            return sigma == 0
                ? tmp.Select(d => (d, d))
                : Core();
            //-----------------------------------------------------------------
            IEnumerable<(double, double)> Core()
            {
                double[] data   = _values;
                double sigmaInv = 1d / sigma;

                for (int i = 0; i < data.Length; ++i)
                    yield return (data[i], this.ZTransformation(data[i], avg, sigmaInv));
            }
        }
        //---------------------------------------------------------------------
        private double ZTransformation(double value, double avg, double sigmaInv) => (value - avg) * sigmaInv;
        //---------------------------------------------------------------------
        private double[] ZTransformationToArrayInternal(double? standardDeviation = null)
        {
            double sigma = standardDeviation ?? this.SampleStandardDeviation;

            if (sigma == 0)
            {
                var tmp = new double[_values.Length];
                _values.CopyTo(tmp, 0);
                return tmp;
            }

            return this.Count < ThresholdForParallel
                ? this.ZTransformationToArraySimd(sigma)
                : this.ZTransformationToArrayParallelizedSimd(sigma);
        }
        //---------------------------------------------------------------------
        internal double[] ZTransformationToArraySimd(double sigma)
        {
            var zTrans = new double[this.Count];

            this.ZTransformationToArrayImpl(zTrans, sigma, 0, _values.Length);

            return zTrans;
        }
        //---------------------------------------------------------------------
        internal double[] ZTransformationToArrayParallelizedSimd(double sigma)
        {
            var zTrans = new double[this.Count];

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range => this.ZTransformationToArrayImpl(zTrans, sigma, range.Item1, range.Item2)
            );

            return zTrans;
        }
        //---------------------------------------------------------------------
        private unsafe void ZTransformationToArrayImpl(double[] zTrans, double sigma, int idxStart, int idxEnd)
        {
            double avg      = this.Mean;
            double sigmaInv = 1d / sigma;

            fixed (double* pSource = _values)
            fixed (double* pTarget = zTrans)
            {
                double* sourceStart   = pSource + idxStart;
                double* sourceCurrent = sourceStart;

                double* targetStart         = pTarget + idxStart;
                double* targetEnd           = pTarget + idxEnd;
                double* targetCurrent       = targetStart;
                double* targetSequentialEnd = default;

                int i = 0;
                int n = idxEnd - idxStart;

                // Only one pointer can be aligned to simd registers.
                // Because writes (stores) have greater penalty for miss-aligned data
                // than reads (loads), the target will be aligned.
                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                    targetSequentialEnd = VectorHelper.GetAlignedPointer(targetStart);
                else
                    targetSequentialEnd = targetEnd;

                // When SIMD is available, first pass is for register alignment.
                // Second pass will be the remaining elements.
            Sequential:
                while (targetCurrent < targetSequentialEnd)
                {
                    *targetCurrent = this.ZTransformation(*sourceCurrent, avg, sigmaInv);
                    sourceCurrent++;
                    targetCurrent++;
                }

                if (Vector.IsHardwareAccelerated)
                {
                    if (targetCurrent >= targetEnd) return;

                    n -= (int)(targetCurrent - targetStart);

                    if (n < Vector<double>.Count)
                    {
                        targetSequentialEnd = targetEnd;
                        goto Sequential;
                    }

                    var avgVec      = new Vector<double>(avg);
                    var sigmaInvVec = new Vector<double>(sigmaInv);

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(sourceCurrent, targetCurrent, 0 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 1 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 2 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 3 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 4 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 5 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 6 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 7 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);

                        sourceCurrent += 8 * Vector<double>.Count;
                        targetCurrent += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(sourceCurrent, targetCurrent, 0 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 1 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 2 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 3 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);

                        sourceCurrent += 4 * Vector<double>.Count;
                        targetCurrent += 4 * Vector<double>.Count;
                        i             += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(sourceCurrent, targetCurrent, 0 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);
                        Core(sourceCurrent, targetCurrent, 1 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);

                        sourceCurrent += 2 * Vector<double>.Count;
                        targetCurrent += 2 * Vector<double>.Count;
                        i             += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(sourceCurrent, targetCurrent, 0 * Vector<double>.Count, avgVec, sigmaInvVec, targetEnd);

                        sourceCurrent += 1 * Vector<double>.Count;
                        targetCurrent += 1 * Vector<double>.Count;
                    }

                    if (targetCurrent < targetEnd)
                    {
                        targetSequentialEnd = targetEnd;
                        goto Sequential;            // second pass for sequential
                    }
                }
            }
            //-----------------------------------------------------------------
            void Core(double* sourceArr, double* targetArr, int offset, Vector<double> avgVec, Vector<double> sigmaInvVec, double* targetEnd)
            {
#if DEBUG_ASSERT
                // targetArr is included -> -1
                Debug.Assert(targetArr + offset + Vector<double>.Count - 1 < targetEnd);
#endif
                Vector<double> vec       = VectorHelper.GetVectorUnaligned(sourceArr + offset);
                Vector<double> zTransVec = (vec - avgVec) * sigmaInvVec;

                // Vector can be written aligned instead of unaligned, because targetArr was aligned in the sequential pass.
                zTransVec.WriteVector(targetArr + offset);
            }
        }
    }
}
