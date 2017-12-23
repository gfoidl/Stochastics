using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        internal const int ThresholdForVariance = 100_000;
        //---------------------------------------------------------------------
        private double CalculateVarianceCore()
        {
            // Threshould determined by benchmark (roughly)
            return this.Count < ThresholdForVariance
                ? this.VarianceCoreSimd()
                : this.VarianceCoreParallelizedSimd();
        }
        //---------------------------------------------------------------------
        internal double VarianceCoreSimd()
        {
            double variance = 0;
            double[] arr    = _values;
            double avg      = this.Mean;
            int i           = 0;

            if (Vector.IsHardwareAccelerated && this.Count >= Vector<double>.Count * 2)
            {
                for (; i < arr.Length - 2 * Vector<double>.Count; i += Vector<double>.Count)
                {
                    var v1 = new Vector<double>(arr, i);
                    var v2 = new Vector<double>(arr, i);
                    variance += Vector.Dot(v1, v2);

                    i += Vector<double>.Count;
                    v1 = new Vector<double>(arr, i);
                    v2 = new Vector<double>(arr, i);
                    variance += Vector.Dot(v1, v2);
                }
            }

            for (; i < arr.Length; ++i)
                variance += arr[i] * arr[i];

            variance -= arr.Length * avg * avg;

            return variance;
        }
        //---------------------------------------------------------------------
        internal double VarianceCoreParallelizedSimd()
        {
            double variance = 0;
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
                        for (; i < range.Item2 - 2 * Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var v1 = new Vector<double>(arr, i);
                            var v2 = new Vector<double>(arr, i);
                            local += Vector.Dot(v1, v2);

                            i += Vector<double>.Count;
                            v1 = new Vector<double>(arr, i);
                            v2 = new Vector<double>(arr, i);
                            local += Vector.Dot(v1, v2);
                        }
                    }

                    for (; i < range.Item2; ++i)
                        local += arr[i] * arr[i];

                    lock (sync) variance += local;
                }
            );

            variance -= this.Count * this.Mean * this.Mean;
            return variance;
        }
    }
}