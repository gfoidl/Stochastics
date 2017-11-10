using System;

namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// Outlier detection based on Chauvenet's criterion.
    /// </summary>
    public class ChauvenetOutlierDetection : OutlierDetection
    {
        private static readonly double _sqrt2Inv = 1d / Math.Sqrt(2);
        //---------------------------------------------------------------------
        /// <summary>
        /// Creates a new instance of <see cref="ChauvenetOutlierDetection" />
        /// </summary>
        /// <param name="sample">
        /// The <see cref="Statistics.Sample" /> on that outlier detection
        /// is performed.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// <paramref name="sample" /> is <c>null</c>.
        /// </exception>
        public ChauvenetOutlierDetection(Sample sample) : base(sample) { }
        //---------------------------------------------------------------------
        /// <summary>
        /// Determines if the <paramref name="value" /> is an outlier or not.
        /// </summary>
        /// <param name="value">The value to check for outlier.</param>
        /// <returns>
        /// <c>true</c> if the value is an outlier, <c>false</c> if the value
        /// is not an outlier.
        /// </returns>
        protected override bool IsOutlier((double Value, double zTransformed) value)
        {
            double tsus        = Math.Abs(value.zTransformed);
            double probOutside = 1 - Erf(tsus * _sqrt2Inv);

            return this.Sample.Count * probOutside < 0.5;
        }
        //---------------------------------------------------------------------
        // Based on https://www.johndcook.com/blog/csharp_erf/
        private static double Erf(double x)
        {
            const double a1 = 0.254829592;
            const double a2 = -0.284496736;
            const double a3 = 1.421413741;
            const double a4 = -1.453152027;
            const double a5 = 1.061405429;
            const double p  = 0.3275911;

            // Save the sign of x
            int sign = Math.Sign(x);
            x        = Math.Abs(x);

            // A&S formula 7.1.26
            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

            return sign * y;
        }
    }
}