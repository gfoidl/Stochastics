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
            var sync     = new object();

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localDelta = this.CalculateDeltaImpl((range.Item1, range.Item2));

                    lock (sync) delta += localDelta;
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

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
                {
                    var avgVec   = new Vector<double>(avg);
                    var deltaVec = new Vector<double>(0);

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        deltaVec += Vector.Abs(vec - avgVec);

                        vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        deltaVec += Vector.Abs(vec - avgVec);
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        delta += deltaVec[j];
                }

                for (; i < n; ++i)
                    delta += Math.Abs(pArray[i] - avg);
            }

            return delta;
        }
    }
}