using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using gfoidl.Stochastics.Partitioners;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        #region IEnumerable
        internal IEnumerable<double> AutoCorrelationSequential()
        {
            double[] arr = _values;
            int n2       = arr.Length / 2;

            for (int m = 0; m < n2; ++m)
            {
                double r_xx = 0;

                for (int k = m; k < arr.Length; ++k)
                    r_xx += arr[k] * arr[k - m];

                yield return r_xx / (arr.Length - m);
            }
        }
        //---------------------------------------------------------------------
        internal IEnumerable<double> AutoCorrelationSimd()
        {
            double[] arr = _values;
            int n2       = arr.Length / 2;

            for (int m = 0; m < n2; ++m)
            {
                double r_xx = 0;
                int k       = m;

                for (; k < arr.Length - 2 * Vector<double>.Count;)
                {
                    var kVec  = new Vector<double>(arr, k);
                    var kmVec = new Vector<double>(arr, k - m);
                    r_xx += Vector.Dot(kVec, kmVec);
                    k += Vector<double>.Count;

                    kVec  = new Vector<double>(arr, k);
                    kmVec = new Vector<double>(arr, k - m);
                    r_xx += Vector.Dot(kVec, kmVec);
                    k += Vector<double>.Count;
                }

                for (; k < arr.Length; ++k)
                    r_xx += arr[k] * arr[k - m];

                yield return r_xx / (arr.Length - m);
            }
        }
        #endregion
        //---------------------------------------------------------------------
        #region ToArray
        internal double[] AutoCorrelationToArraySimd()
        {
            int n    = _values.Length;
            int n2   = n / 2;
            var corr = new double[n2];

            this.AutoCorrelationToArrayImpl(corr, (0, n2));

            return corr;
        }
        //---------------------------------------------------------------------
        internal double[] AutoCorrelationToArrayParallelSimd()
        {
            var corr                      = new double[_values.Length / 2];
            int n                         = _values.Length;
            var parallelOptions           = GetParallelOptions();
            const int partitionMultiplier = 8;
            int partitionCount            = parallelOptions.MaxDegreeOfParallelism * partitionMultiplier;

            Parallel.ForEach(
                WorkloadPartitioner.Create(n / 2, loadFactorAtStart: n, loadFactorAtEnd: n / 2, partitionCount),
                parallelOptions,
                range => this.AutoCorrelationToArrayImpl(corr, range)
            );

            return corr;
        }
        //---------------------------------------------------------------------
        private unsafe void AutoCorrelationToArrayImpl(double[] corr, (int Start, int End) range)
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            fixed (double* pCorr  = corr)
            {
                for (int m = range.Start; m < range.End; ++m)
                {
                    double r_xx = 0;
                    int k       = m;

                    if (Vector.IsHardwareAccelerated && (n - m) >= Vector<double>.Count * 2)
                    {
                        double* a_k  = &pArray[k];
                        double* a_km = pArray;

                        for (; k < n - 2 * Vector<double>.Count; k += 2 * Vector<double>.Count)
                        {
                            Vector<double> kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            Vector<double> kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);

                            kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);
                        }
                    }

                    for (; k < n; ++k)
                        r_xx += pArray[k] * pArray[k - m];

                    pCorr[m] = r_xx / (n - m);
                }
            }
        }
        #endregion
    }
}