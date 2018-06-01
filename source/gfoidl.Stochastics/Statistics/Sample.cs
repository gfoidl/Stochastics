using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using gfoidl.Stochastics.Enumerators;

namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// Represents position, scatter and shape parameters of sample data.
    /// The parameters are lazy-evaluated.
    /// </summary>
    /// <remarks>
    /// Computation of statistical parameter is done in parallel, when
    /// <see cref="Count" /> is greater or equal <see cref="SampleThresholds.ThresholdForParallel" />.
    /// SIMD is used where possible and where it brings an advantage.
    /// </remarks>
    public partial class Sample
    {
        private readonly double[] _values;
        //---------------------------------------------------------------------
        /// <summary>
        /// Sample.
        /// </summary>
        public ArrayEnumerable<double> Values => _values;
        //---------------------------------------------------------------------
        /// <summary>
        /// Sample size.
        /// </summary>
        public int Count => _values.Length;
        //---------------------------------------------------------------------
        /// <summary>
        /// Creates a new instance of <see cref="Sample" />
        /// </summary>
        /// <param name="values">The sample, which gets analyzed (lazy evaluated).</param>
        /// <exception cref="ArgumentNullException"><paramref name="values" /> is <c>null</c>.</exception>
        public Sample(IEnumerable<double> values)
        {
            if (values == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.values);

            if (values is double[] tmp)
                _values = tmp;
            else
                _values = values.ToArray();
        }
        //---------------------------------------------------------------------
#pragma warning disable CS1591
        public static implicit operator Sample(double[] values)     => new Sample(values);
        public static implicit operator Sample(List<double> values) => new Sample(values);
#pragma warning restore CS1591
        //---------------------------------------------------------------------
        private double[] _sortedValues;
        /// <summary>
        /// Sample data, sorted ascending.
        /// </summary>
        public ICollection<double> SortedValues
        {
            get
            {
                if (_sortedValues == null)
                {
                    _sortedValues = new double[_values.Length];
                    _values.CopyTo(_sortedValues, 0);
                    Array.Sort(_sortedValues);
                }

                return _sortedValues;
            }
        }
        //---------------------------------------------------------------------
        private double _mean = double.NaN;
        /// <summary>
        /// Mean / arithmetic average.
        /// </summary>
        public double Mean
        {
            get
            {
                if (double.IsNaN(_mean))
                    this.CalculateAverageAndVarianceCore();

                return _mean;
            }
            internal set => _mean = value;
        }
        //---------------------------------------------------------------------
        private double _median = double.NaN;
        /// <summary>
        /// Median.
        /// </summary>
        public double Median
        {
            get
            {
                if (double.IsNaN(_median))
                    _median = this.CalculateMedian();

                return _median;
            }
        }
        //---------------------------------------------------------------------
        private double _max = double.NaN;
        /// <summary>
        /// Max.
        /// </summary>
        public double Max
        {
            get
            {
                if (double.IsNaN(_max))
                    this.GetMinMax();

                return _max;
            }
            internal set => _max = value;
        }
        //---------------------------------------------------------------------
        private double _min = double.NaN;
        /// <summary>
        /// Min.
        /// </summary>
        public double Min
        {
            get
            {
                if (double.IsNaN(_min))
                    this.GetMinMax();

                return _min;
            }
            internal set => _min = value;
        }
        //---------------------------------------------------------------------
        private double _range = double.NaN;
        /// <summary>
        /// Range.
        /// </summary>
        public double Range
        {
            get
            {
                if (double.IsNaN(_range))
                    _range = this.Max - this.Min;

                return _range;
            }
        }
        //---------------------------------------------------------------------
        private double _delta = double.NaN;
        /// <summary>
        /// Delta -- mean absolute deviation.
        /// </summary>
        public double Delta
        {
            get
            {
                if (double.IsNaN(_delta))
                    _delta = this.CalculateDelta();

                return _delta;
            }
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Points to <see cref="SampleStandardDeviation" />.
        /// </summary>
        public double Sigma => this.SampleStandardDeviation;
        //---------------------------------------------------------------------
        private double _sigma = double.NaN;
        /// <summary>
        /// Standard deviation (scattering). 1/N
        /// </summary>
        public double StandardDeviation
        {
            get
            {
                if (double.IsNaN(_sigma))
                    _sigma = Math.Sqrt(this.Variance);

                return _sigma;
            }
        }
        //---------------------------------------------------------------------
        private double _sampleSigma = double.NaN;
        /// <summary>
        /// Sample standard deviation. 1/(N-1)
        /// </summary>
        public double SampleStandardDeviation
        {
            get
            {
                if (double.IsNaN(_sampleSigma))
                    _sampleSigma = Math.Sqrt(this.SampleVariance);

                return _sampleSigma;
            }
        }
        //---------------------------------------------------------------------
        private double _variance = double.NaN;
        /// <summary>
        /// Variance. 1/N
        /// </summary>
        public double Variance
        {
            get
            {
                if (double.IsNaN(_variance))
                    _variance = this.CalculateVariance();

                return _variance;
            }
        }
        //---------------------------------------------------------------------
        private double _sampleVariance = double.NaN;
        /// <summary>
        /// Sample Variance. 1/(N-1)
        /// </summary>
        public double SampleVariance
        {
            get
            {
                if (double.IsNaN(_sampleVariance))
                    _sampleVariance = this.CalculateSampleVariance();

                return _sampleVariance;
            }
        }
        //---------------------------------------------------------------------
        private double _skewness = double.NaN;
        /// <summary>
        /// Skewness.
        /// </summary>
        /// <remarks>
        /// Skewness is a measure of the asymmetry of the probability distribution
        /// about its mean. The skewness value can be positive or negative, or undefined.
        /// <para>
        /// negative skew: The left tail is longer; the mass of the distribution is 
        /// concentrated on the right of the figure. The distribution is said to be 
        /// left-skewed, left-tailed, or skewed to the left, despite the fact that the 
        /// curve itself appears to be skewed or leaning to the right; left instead 
        /// refers to the left tail being drawn out and, often, the mean being skewed 
        /// to the left of a typical center of the data.
        /// </para>
        /// <para>
        /// positive skew: The right tail is longer; the mass of the distribution is 
        /// concentrated on the left of the figure. The distribution is said to be 
        /// right-skewed, right-tailed, or skewed to the right, despite the fact that the 
        /// curve itself appears to be skewed or leaning to the left; right instead 
        /// refers to the right tail being drawn out and, often, the mean being skewed 
        /// to the right of a typical center of the data.
        /// </para>
        /// </remarks>
        /// <seealso cref="!:https://en.wikipedia.org/wiki/Skewness" />
        public double Skewness
        {
            get
            {
                if (double.IsNaN(_skewness))
                    this.CalculateSkewnessAndKurtosis();

                return _skewness;
            }
        }
        //---------------------------------------------------------------------
        private double _kurtosis = double.NaN;
        /// <summary>
        /// Kurtosis
        /// </summary>
        /// <remarks>
        /// A measure of the curvature of a distribution. The value of 3 corresponds
        /// the ideal normal distribution. A value lower than 3 indicates one flatter 
        /// distribution than the normal distribution, whereas a steeper distribution 
        /// corresponding to a value greater than 3.
        /// </remarks>
        public double Kurtosis
        {
            get
            {
                if (double.IsNaN(_kurtosis))
                    this.CalculateSkewnessAndKurtosis();

                return _kurtosis;
            }
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Returns the z-Transformed sample.
        /// </summary>
        /// <param name="standardDeviation">
        /// The standard deviation used to perform the z-Transformation.
        /// Can bei either <see cref="StandardDeviation" /> or <see cref="SampleStandardDeviation" />
        /// depending on the sample and on what to inspect.
        /// <para>
        /// When no value is given, <see cref="SampleStandardDeviation" /> will be used.
        /// </para>
        /// </param>
        /// <returns>The z-Transformed sample.</returns>
        /// <seealso cref="!:https://en.wikipedia.org/wiki/Standard_score" />
        public IEnumerable<double> ZTransformation(double? standardDeviation = null)
            => this.ZTransformationInternal(standardDeviation).Select(t => t.zTransformed);
        //---------------------------------------------------------------------
        /// <summary>
        /// Returns the z-Transformed sample.
        /// </summary>
        /// <param name="standardDeviation">
        /// The standard deviation used to perform the z-Transformation.
        /// Can bei either <see cref="StandardDeviation" /> or <see cref="SampleStandardDeviation" />
        /// depending on the sample and on what to inspect.
        /// <para>
        /// When no value is given, <see cref="SampleStandardDeviation" /> will be used.
        /// </para>
        /// </param>
        /// <returns>The z-Transformed sample.</returns>
        /// <seealso cref="!:https://en.wikipedia.org/wiki/Standard_score" />
        public double[] ZTransformationToArray(double? standardDeviation = null)
            => this.ZTransformationToArrayInternal(standardDeviation);
        //---------------------------------------------------------------------
        /// <summary>
        /// Autocorrelation
        /// </summary>
        /// <returns>The autocorrelation of the sample.</returns>
        public IEnumerable<double> AutoCorrelation() => Vector.IsHardwareAccelerated
            ? this.AutoCorrelationSimd()
            : this.AutoCorrelationSequential();
        //---------------------------------------------------------------------
        /// <summary>
        /// Autocorrelation
        /// </summary>
        /// <returns>The autocorrelation of the sample.</returns>
        /// <remarks>
        /// When <see cref="Count" /> is greater or equal <see cref="SampleThresholds.ThresholdForAutocorrelationParallel"/>
        /// all CPUs are used for processing.
        /// </remarks>
        public double[] AutoCorrelationToArray() => this.Count < SampleThresholds.ThresholdForAutocorrelationParallel
            ? this.AutoCorrelationToArraySimd()
            : this.AutoCorrelationToArrayParallelSimd();
        //---------------------------------------------------------------------
        /// <summary>Returns a string that represents the current object.</summary>
        /// <returns>A string that represents the current object.</returns>
        public override string ToString()
        {
            var sb = new StringBuilder();

            foreach(var pi in typeof(Sample).GetProperties())
            {
                if (pi.Name == nameof(this.Values) || pi.Name == nameof(this.SortedValues) || pi.Name == "Item")
                    continue;

                sb.AppendFormat("{0,-23}", pi.Name).Append(": ").Append(pi.GetValue(this)).AppendLine();
            }

            return sb.ToString();
        }
    }
}
