using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private void CalculateAverageAndVarianceCore()
        {
            double avg      = 0;
            double variance = 0;

            if (this.Count < ThresholdForParallel)
                this.CalculateAverageAndVarianceCoreSimd(out avg, out variance);
            else
                this.CalculateAverageAndVarianceCoreParallelizedSimd(out avg, out variance);

            avg      /= this.Count;
            variance -= this.Count * avg * avg;

            _mean         = avg;
            _varianceCore = variance;
        }
        //---------------------------------------------------------------------
        internal void CalculateAverageAndVarianceCoreSimd(out double avg, out double variance)
        {
            this.CalculateAverageAndVarianceCoreImpl(0, this.Count, out avg, out variance);
        }
        //---------------------------------------------------------------------
        internal void CalculateAverageAndVarianceCoreParallelizedSimd(out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range =>
                {
                    this.CalculateAverageAndVarianceCoreImpl(range.Item1, range.Item2, out double localAvg, out double localVariance);
                    localAvg.SafeAdd(ref tmpAvg);
                    localVariance.SafeAdd(ref tmpVariance);
                }
            );

            avg      = tmpAvg;
            variance = tmpVariance;
        }
        //---------------------------------------------------------------------
        private unsafe void CalculateAverageAndVarianceCoreImpl(int i, int n, out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec = Vector<double>.Zero;

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec, ref tmpVariance);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref tmpVariance);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref tmpVariance);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance);

                        i += Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    tmpAvg += avgVec.ReduceSum();
                }

                for (; i < n; ++i)
                {
                    tmpAvg      += pArray[i];
                    tmpVariance += pArray[i] * pArray[i];
                }

                avg      = tmpAvg;
                variance = tmpVariance;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec, ref double var)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec += vec;
                var    += Vector.Dot(vec, vec);
            }
        }
    }
}
