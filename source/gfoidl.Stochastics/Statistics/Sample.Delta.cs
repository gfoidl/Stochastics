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
        internal unsafe double CalculateDeltaSimd()
        {
            double delta = 0;
            double avg   = this.Mean;
            int n        = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
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

            return delta / n;
        }
        //---------------------------------------------------------------------
        internal unsafe double CalculateDeltaParallelizedSimd()
        {
            double delta = 0;
            var sync     = new object();

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double localDelta = 0;
                    double avg        = this.Mean;
                    int n             = range.Item2;

                    fixed (double* pArray = _values)
                    {
                        int i       = range.Item1;
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
                                localDelta += deltaVec[j];
                        }

                        for (; i < n; ++i)
                            localDelta += Math.Abs(pArray[i] - avg);
                    }

                    lock (sync) delta += localDelta;
                }
            );

            return delta / _values.Length;
        }
    }
}