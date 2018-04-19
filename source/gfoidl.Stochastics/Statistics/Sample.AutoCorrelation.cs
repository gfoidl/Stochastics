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

                    if (Vector.IsHardwareAccelerated && (n - m) >= Vector<double>.Count)
                    {
                        double* a_k  = &pArray[k];
                        double* a_km = pArray;

                        for (; k < n - 8 * Vector<double>.Count; k += 8 * Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 1 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 2 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 3 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 4 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 5 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 6 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 7 * Vector<double>.Count, ref r_xx);

                            a_k  += 8 * Vector<double>.Count;
                            a_km += 8 * Vector<double>.Count;
                        }

                        if (k < n - 4 * Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 1 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 2 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 3 * Vector<double>.Count, ref r_xx);

                            a_k  += 4 * Vector<double>.Count;
                            a_km += 4 * Vector<double>.Count;
                            k    += 4 * Vector<double>.Count;
                        }

                        if (k < n - 2 * Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 1 * Vector<double>.Count, ref r_xx);

                            a_k  += 2 * Vector<double>.Count;
                            a_km += 2 * Vector<double>.Count;
                            k    += 2 * Vector<double>.Count;
                        }

                        if (k < n - Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);

                            k += Vector<double>.Count;
                        }
                    }

                    for (; k < n; ++k)
                        r_xx += pArray[k] * pArray[k - m];

                    pCorr[m] = r_xx / (n - m);
                }
            }
            //-----------------------------------------------------------------
            void Core(double* a_k, double* a_km, int offset, ref double r_xx)
            {
                Vector<double> kVec  = VectorHelper.GetVector(a_k + offset);
                Vector<double> kmVec = VectorHelper.GetVector(a_km + offset);
                r_xx += Vector.Dot(kVec, kmVec);
            }
        }
        #endregion
    }
}
