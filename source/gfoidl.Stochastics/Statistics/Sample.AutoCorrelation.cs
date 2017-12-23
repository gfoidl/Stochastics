using System.Collections.Generic;
using System.Numerics;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        internal IEnumerable<double> AutoCorrelationSimd()
        {
            double[] arr = _values;
            int n2       = arr.Length >> 1;

            for (int m = 0; m < n2; ++m)
            {
                double r_xx = 0;
                int k       = m;

                for (; k < arr.Length - 2 * Vector<double>.Count; k += Vector<double>.Count)
                {
                    var kVec  = new Vector<double>(arr, k);
                    var kmVec = new Vector<double>(arr, k - m);
                    r_xx += Vector.Dot(kVec, kmVec);

                    k += Vector<double>.Count;
                    kVec  = new Vector<double>(arr, k);
                    kmVec = new Vector<double>(arr, k - m);
                    r_xx += Vector.Dot(kVec, kmVec);
                }

                for (; k < arr.Length; ++k)
                    r_xx += arr[k] * arr[k - m];

                yield return r_xx / (arr.Length - m);
            }
        }
        //---------------------------------------------------------------------
        internal IEnumerable<double> AutoCorrelationSequential()
        {
            double[] arr = _values;
            int n2       = arr.Length >> 1;

            for (int m = 0; m < n2; ++m)
            {
                double r_xx = 0;

                for (int k = m; k < arr.Length; ++k)
                    r_xx += arr[k] * arr[k - m];

                yield return r_xx / (arr.Length - m);
            }
        }
    }
}