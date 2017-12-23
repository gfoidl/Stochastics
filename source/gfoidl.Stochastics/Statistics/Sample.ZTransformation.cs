using System.Collections.Generic;
using System.Linq;

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
    }
}