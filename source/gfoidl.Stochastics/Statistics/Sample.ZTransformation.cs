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
            // TODO: unsafe SIMD
            var zTrans      = new double[this.Count];
            double[] arr    = _values;
            double avg      = this.Mean;
            double sigmaInv = 1d / sigma;
            int i           = 0;

            if (Vector.IsHardwareAccelerated && this.Count >= Vector<double>.Count * 2)
            {
                var avgVec      = new Vector<double>(avg);
                var sigmaInvVec = new Vector<double>(sigmaInv);

                for (; i < arr.Length - 2 * Vector<double>.Count; i += Vector<double>.Count)
                {
                    var vec       = new Vector<double>(arr, i);
                    var zTransVec = (vec - avgVec) * sigmaInvVec;
                    zTransVec.CopyTo(zTrans, i);

                    i += Vector<double>.Count;
                    vec = new Vector<double>(arr, i);
                    zTransVec = (vec - avgVec) * sigmaInvVec;
                    zTransVec.CopyTo(zTrans, i);
                }
            }

            for (; i < arr.Length; ++i)
                zTrans[i] = this.ZTransformation(arr[i], avg, sigmaInv);

            return zTrans;
        }
        //---------------------------------------------------------------------
        internal double[] ZTransformationToArrayParallelizedSimd(double sigma)
        {
            // TODO: unsafe SIMD
            var zTrans = new double[this.Count];

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double[] arr    = _values;
                    double avg      = this.Mean;
                    double sigmaInv = 1d / sigma;
                    int i           = range.Item1;

                    // RCE
                    if ((uint)range.Item1 >= arr.Length || (uint)range.Item2 > arr.Length
                        || (uint)range.Item1 >= zTrans.Length || (uint)range.Item2 > zTrans.Length)
                        ThrowHelper.ThrowArgumentOutOfRange(nameof(range));

                    if (Vector.IsHardwareAccelerated && (range.Item2 - range.Item1) >= Vector<double>.Count * 2)
                    {
                        var avgVec      = new Vector<double>(avg);
                        var sigmaInvVec = new Vector<double>(sigmaInv);

                        for (; i < range.Item2 - 2 * Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var vec       = new Vector<double>(arr, i);
                            var zTransVec = (vec - avgVec) * sigmaInvVec;
                            zTransVec.CopyTo(zTrans, i);

                            i += Vector<double>.Count;
                            vec = new Vector<double>(arr, i);
                            zTransVec = (vec - avgVec) * sigmaInvVec;
                            zTransVec.CopyTo(zTrans, i);
                        }
                    }

                    for (; i < range.Item2; ++i)
                        zTrans[i] = this.ZTransformation(arr[i], avg, sigmaInv);
                }
            );

            return zTrans;
        }
    }
}