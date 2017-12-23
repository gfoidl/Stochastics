using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        internal const int ThresholdForDelta = 100_000;
        //---------------------------------------------------------------------
        private double CalculateDelta()
        {
            // Threshould determined by benchmark (roughly)
            return this.Count < ThresholdForDelta
                ? this.CalculateDeltaSimd()
                : this.CalculateDeltaParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double CalculateDeltaSimd()
        {
            double delta = 0;
            double[] arr = _values;
            double avg   = this.Mean;
            int i        = 0;

            if (Vector.IsHardwareAccelerated && this.Count >= Vector<double>.Count * 2)
            {
                var avgVec   = new Vector<double>(avg);
                var deltaVec = new Vector<double>(0);

                for (; i < arr.Length - 2 * Vector<double>.Count; i += Vector<double>.Count)
                {
                    var vec = new Vector<double>(arr, i);
                    deltaVec += Vector.Abs(vec - avgVec);

                    i += Vector<double>.Count;
                    vec = new Vector<double>(arr, i);
                    deltaVec += Vector.Abs(vec - avgVec);
                }

                for (int j = 0; j < Vector<double>.Count; ++j)
                    delta += deltaVec[j];
            }

            for (; i < arr.Length; ++i)
                delta += Math.Abs(arr[i] - avg);

            return delta / arr.Length;
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
                    double localDelta = 0;
                    double[] arr      = _values;
                    double avg        = this.Mean;
                    int i             = range.Item1;

                    // RCE
                    if ((uint)range.Item1 >= arr.Length || (uint)range.Item2 > arr.Length)
                        ThrowHelper.ThrowArgumentOutOfRange(nameof(range));

                    if (Vector.IsHardwareAccelerated && (range.Item2 - range.Item1) >= Vector<double>.Count * 2)
                    {
                        var avgVec   = new Vector<double>(avg);
                        var deltaVec = new Vector<double>(0);

                        for (; i < range.Item2 - 2 * Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var arrVec = new Vector<double>(arr, i);
                            deltaVec += Vector.Abs(arrVec - avgVec);

                            i += Vector<double>.Count;
                            arrVec = new Vector<double>(arr, i);
                            deltaVec += Vector.Abs(arrVec - avgVec);
                        }

                        for (int j = 0; j < Vector<double>.Count; ++j)
                            localDelta += deltaVec[j];
                    }

                    for (; i < range.Item2; ++i)
                        localDelta += Math.Abs(arr[i] - avg);

                    lock (sync) delta += localDelta;
                }
            );

            return delta / _values.Length;
        }
    }
}