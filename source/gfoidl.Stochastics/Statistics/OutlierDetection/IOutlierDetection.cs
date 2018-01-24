using System.Collections.Generic;

namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// Outlier detection
    /// </summary>
    public interface IOutlierDetection
    {
        /// <summary>
        /// The <see cref="Statistics.Sample" /> on that outlier detection
        /// is performed.
        /// </summary>
        Sample Sample { get; }
        //---------------------------------------------------------------------
        /// <summary>
        /// Gets the outliers of the <see cref="Sample" />.
        /// </summary>
        /// <returns>The outliers of the <see cref="Sample" />.</returns>
        IEnumerable<double> GetOutliers();
        //---------------------------------------------------------------------
        /// <summary>
        /// Gets the values of the <see cref="Sample" /> without outliers.
        /// </summary>
        /// <returns>The values of the <see cref="Sample" /> without outliers.</returns>
        IEnumerable<double> GetValuesWithoutOutliers();
    }
}
