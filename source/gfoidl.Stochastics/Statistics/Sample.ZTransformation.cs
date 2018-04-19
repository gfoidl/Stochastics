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
        private unsafe void ZTransformationToArrayImpl(double[] zTrans, double sigma, int i, int n)
        {
            double avg      = this.Mean;
            double sigmaInv = 1d / sigma;

            fixed (double* pSource = _values)
            fixed (double* pTarget = zTrans)
            {
                double* source = pSource + i;
                double* target = pTarget + i;
                n             -= i;
                i              = 0;
                double* end    = source + n;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec      = new Vector<double>(avg);
                    var sigmaInvVec = new Vector<double>(sigmaInv);

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 1 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 2 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 3 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 4 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 5 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 6 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 7 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 8 * Vector<double>.Count;
                        target += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 1 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 2 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 3 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 4 * Vector<double>.Count;
                        target += 4 * Vector<double>.Count;
                        i      += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 1 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 2 * Vector<double>.Count;
                        target += 2 * Vector<double>.Count;
                        i      += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 1 * Vector<double>.Count;
                        target += 1 * Vector<double>.Count;
                    }
                }

                while (source < end)
                {
                    *target = this.ZTransformation(*source, avg, sigmaInv);
                    source++;
                    target++;
                }
            }
            //-----------------------------------------------------------------
            void Core(double* sourceArr, double* targetArr, int offset, Vector<double> avgVec, Vector<double> sigmaInvVec)
            {
                Vector<double> vec       = VectorHelper.GetVector(sourceArr + offset);
                Vector<double> zTransVec = (vec - avgVec) * sigmaInvVec;
                zTransVec.WriteVector(targetArr + offset);
            }
        }
    }
}
