using System;
using System.Collections.Generic;
using System.Linq;

namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// Base class for outlier detection.
    /// </summary>
    public abstract class OutlierDetection : IOutlierDetection
    {
        /// <summary>
        /// The <see cref="Statistics.Sample" /> on that outlier detection
        /// is performed.
        /// </summary>
        public Sample Sample { get; }
        //---------------------------------------------------------------------
        /// <summary>
        /// Creates a new instance of <see cref="OutlierDetection" />
        /// </summary>
        /// <param name="sample">
        /// The <see cref="Statistics.Sample" /> on that outlier detection
        /// is performed.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// <paramref name="sample" /> is <c>null</c>.
        /// </exception>
        protected OutlierDetection(Sample sample)
        {
            this.Sample = sample ?? throw new ArgumentNullException(nameof(sample));
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Gets the outliers of the <see cref="Sample" />.
        /// </summary>
        /// <returns>The outliers of the <see cref="Sample" />.</returns>
        public IEnumerable<double> GetOutliers() =>
            this.Sample.ZTransformationInternal()
                       .Where(this.IsOutlier)
                       .Select(t => t.Value);
        //---------------------------------------------------------------------
        /// <summary>
        /// Gets the values of the <see cref="Sample" /> without outliers.
        /// </summary>
        /// <returns>The values of the <see cref="Sample" /> without outliers.</returns>
        public IEnumerable<double> GetValuesWithoutOutliers() =>
            this.Sample.ZTransformationInternal()
                       .Where(t => !this.IsOutlier(t))
                       .Select(t => t.Value);
        //---------------------------------------------------------------------
        /// <summary>
        /// Determines if the <paramref name="value" /> is an outlier or not.
        /// </summary>
        /// <param name="value">The value to check for outlier.</param>
        /// <returns>
        /// <c>true</c> if the value is an outlier, <c>false</c> if the value
        /// is not an outlier.
        /// </returns>
        protected abstract bool IsOutlier((double Value, double zTransformed) value);
    }
}