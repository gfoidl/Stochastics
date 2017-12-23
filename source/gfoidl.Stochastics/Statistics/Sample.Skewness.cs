﻿using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        internal const int ThreshouldForSkewness = 100_000;
        //---------------------------------------------------------------------
        private double CalculateSkewness()
        {
            // Threshould set same as by Delta and VarianceCore.
            // Assuming it will be similar.
            return this.Count < ThreshouldForSkewness
                ? this.CalculateSkewnessSimd()
                : this.CalculateSkewnessParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double CalculateSkewnessSimd()
        {
            double skewness = 0;
            double[] arr    = _values;
            double avg      = this.Mean;
            double sigma    = this.StandardDeviation;
            int i           = 0;

            if (Vector.IsHardwareAccelerated && this.Count >= Vector<double>.Count * 2)
            {
                var avgVec  = new Vector<double>(avg);
                var skewVec = new Vector<double>(0);

                for (; i < arr.Length - 2 * Vector<double>.Count; i += Vector<double>.Count)
                {
                    var vec = new Vector<double>(arr, i);
                    vec -= avgVec;
                    skewVec += vec * vec * vec;

                    i += Vector<double>.Count;
                    vec = new Vector<double>(arr, i);
                    vec -= avgVec;
                    skewVec += vec * vec * vec;
                }

                for (int j = 0; j < Vector<double>.Count; ++j)
                    skewness += skewVec[j];
            }

            for (; i < arr.Length; ++i)
            {
                double t = arr[i] - avg;
                skewness += t * t * t;
            }

            skewness /= arr.Length * sigma * sigma * sigma;

            return skewness;
        }
        //---------------------------------------------------------------------
        internal double CalculateSkewnessParallelizedSimd()
        {
            double skewness = 0;
            var sync        = new object();

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    double local = 0;
                    double[] arr = _values;
                    double avg   = this.Mean;
                    int i        = range.Item1;

                    // RCE
                    if ((uint)range.Item1 >= arr.Length || (uint)range.Item2 > arr.Length)
                        ThrowHelper.ThrowArgumentOutOfRange(nameof(range));

                    if (Vector.IsHardwareAccelerated && (range.Item2 - range.Item1) >= Vector<double>.Count * 2)
                    {
                        var avgVec  = new Vector<double>(avg);
                        var skewVec = new Vector<double>(0);

                        for (; i < range.Item2 - 2 * Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var vec = new Vector<double>(arr, i);
                            vec -= avgVec;
                            skewVec += vec * vec * vec;

                            i += Vector<double>.Count;
                            vec = new Vector<double>(arr, i);
                            vec -= avgVec;
                            skewVec += vec * vec * vec;
                        }

                        for (int j = 0; j < Vector<double>.Count; ++j)
                            local += skewVec[j];
                    }

                    for (; i < range.Item2; ++i)
                    {
                        double t = arr[i] - avg;
                        local += t * t * t;
                    }

                    lock (sync) skewness += local;
                }
            );

            double sigma = this.StandardDeviation;
            skewness /= _values.Length * sigma * sigma * sigma;
            return skewness;
        }
    }
}