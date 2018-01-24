using System;
using static gfoidl.Stochastics.SpecialFunctions;

namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// Outlier detection based on Chauvenet's criterion.
    /// </summary>
    public class ChauvenetOutlierDetection : OutlierDetection
    {
        private double[] _tsusSqrt2Inv;
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
            //double probOutside = 1 - Erf(tsus * _sqrt2Inv);
            // Erfc = 1 - Erf -> so use this :-)
            double probOutside = Erfc(tsus * _sqrt2Inv);

            return this.Sample.Count * probOutside < 0.5;
        }
    }
}
