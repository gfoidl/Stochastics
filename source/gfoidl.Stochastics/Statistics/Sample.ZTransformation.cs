using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

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

            this.ZTransformationToArrayImpl(zTrans, sigma, (0, _values.Length));

            return zTrans;
        }
        //---------------------------------------------------------------------
        internal double[] ZTransformationToArrayParallelizedSimd(double sigma)
        {
            var zTrans = new double[this.Count];

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range => this.ZTransformationToArrayImpl(zTrans, sigma, (range.Item1, range.Item2))
            );

            return zTrans;
        }
        //---------------------------------------------------------------------
        private unsafe void ZTransformationToArrayImpl(double[] zTrans, double sigma, (int start, int end) range)
        {
            double avg      = this.Mean;
            double sigmaInv = 1d / sigma;
            var (i, n)      = range;

            fixed (double* pSource = _values)
            fixed (double* pTarget = zTrans)
            {
                double* sourceArr = pSource + i;
                double* targetArr = pTarget + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
                {
                    var avgVec      = new Vector<double>(avg);
                    var sigmaInvVec = new Vector<double>(sigmaInv);

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> vec       = VectorHelper.GetVectorWithAdvance(ref sourceArr);
                        Vector<double> zTransVec = (vec - avgVec) * sigmaInvVec;
                        zTransVec.WriteVectorWithAdvance(ref targetArr);

                        vec       = VectorHelper.GetVectorWithAdvance(ref sourceArr);
                        zTransVec = (vec - avgVec) * sigmaInvVec;
                        zTransVec.WriteVectorWithAdvance(ref targetArr);
                    }
                }

                for (; i < n; ++i)
                    pTarget[i] = this.ZTransformation(pSource[i], avg, sigmaInv);
            }
        }
    }
}