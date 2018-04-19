using System.Collections.Generic;
using System.Runtime.CompilerServices;
using gfoidl.Stochastics.Builders;
using gfoidl.Stochastics.Enumerators;

namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// A builder for <see cref="Sample" />.
    /// </summary>
    public class SampleBuilder
    {
        // Must not be readonly, cf. https://gist.github.com/gfoidl/14b07dfe8ee5cb093f216f8a85759d88
        private ArrayBuilder<double> _arrayBuilder;
        private double               _min = double.MaxValue;
        private double               _max = double.MinValue;
        private double               _sum;
        private bool                 _canUseStats = true;
        //---------------------------------------------------------------------
        /// <summary>
        /// Creates a new instance of <see cref="SampleBuilder" />.
        /// </summary>
        public SampleBuilder() => _arrayBuilder = new ArrayBuilder<double>(true);
        //---------------------------------------------------------------------
        /// <summary>
        /// Adds an <paramref name="item" /> to the builder.
        /// </summary>
        /// <param name="item">The value to add.</param>
        public void Add(double item) => this.AddCore(item, ref _min, ref _max, ref _sum);
        //---------------------------------------------------------------------
        /// <summary>
        /// Adds an enumeration to the builder.
        /// </summary>
        /// <param name="values">The items to add.</param>
        /// <remarks>
        /// This methods uses a "short-cut" for arrays.
        /// </remarks>
        /// <exception cref="System.ArgumentNullException">
        /// <paramref name="values" /> is <c>null</c>.
        /// </exception>
        public void Add(IEnumerable<double> values)
        {
            if (values == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.values);

            if (values is double[] array)
            {
                _arrayBuilder.AddRange(array);
                _canUseStats = false;
                return;
            }

            double min = double.MaxValue;
            double max = double.MinValue;
            double sum = 0;

            foreach (double item in values)
                this.AddCore(item, ref min, ref max, ref sum);

            if (_canUseStats)
            {
                _min = min;
                _max = max;
                _sum = sum;
            }
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Adds an enumeration to the builder, and "echos" the added values.
        /// </summary>
        /// <param name="values">The items to add.</param>
        /// <returns>The added values.</returns>
        /// <remarks>
        /// This methods uses a "short-cut" for arrays. When an array should be
        /// added, a <see cref="ArrayEnumerable{T}" /> is returned.
        /// </remarks>
        /// <exception cref="System.ArgumentNullException">
        /// <paramref name="values" /> is <c>null</c>.
        /// </exception>
        public IEnumerable<double> AddWithYield(IEnumerable<double> values)
        {
            if (values == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.values);

            if (values is double[] array)
                return this.AddWithYield(array);

            return Core();
            //-----------------------------------------------------------------
            IEnumerable<double> Core()
            {
                double min = double.MaxValue;
                double max = double.MinValue;
                double sum = 0;

                foreach (double item in values)
                {
                    this.AddCore(item, ref min, ref max, ref sum);
                    yield return item;
                }

                if (_canUseStats)
                {
                    _min = min;
                    _max = max;
                    _sum = sum;
                }
            }
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Adds an array to the builder, and "echos" the added values.
        /// </summary>
        /// <param name="array">The items to add.</param>
        /// <returns>The added values.</returns>
        /// <exception cref="System.ArgumentNullException">
        /// <paramref name="array" /> is <c>null</c>.
        /// </exception>
        public ArrayEnumerable<double> AddWithYield(double[] array)
        {
            if (array == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.array);

            _arrayBuilder.AddRange(array);
            _canUseStats = false;

            return new ArrayEnumerable<double>(array);
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void AddCore(double item, ref double min, ref double max, ref double sum)
        {
            _arrayBuilder.Add(item);

            if (_canUseStats)
            {
                sum += item;
                if (item < min) min = item;
                if (item > max) max = item;
            }
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Builds and returns the <see cref="Sample" />.
        /// </summary>
        /// <returns>The <see cref="Sample" /> built by this builder.</returns>
        public Sample GetSample()
        {
            ref var arrayBuilder = ref _arrayBuilder;

            double[] values = arrayBuilder.ToArray();

            var sample = new Sample(values);

            if (_canUseStats)
            {
                sample.Min  = _min;
                sample.Max  = _max;
                sample.Mean = _sum / arrayBuilder.Count;
            }

            return sample;
        }
    }
    //-------------------------------------------------------------------------
    /// <summary>
    /// Extension-methods for <see cref="SampleBuilder" />.
    /// </summary>
    public static class SampleBuilderExtensions
    {
        /// <summary>
        /// Adds an enumeration to the builder, and "echos" the added values.
        /// </summary>
        /// <param name="values">The values to add to the builder.</param>
        /// <param name="sampleBuilder">The builder, to which the values are added.</param>
        /// <returns>The added values.</returns>
        /// <exception cref="System.ArgumentNullException">
        /// <paramref name="values" /> is <c>null</c>.
        /// </exception>
        /// <exception cref="System.ArgumentNullException">
        /// <paramref name="sampleBuilder" /> is <c>null</c>.
        /// </exception>
        public static IEnumerable<double> AddToSampleBuilder(this IEnumerable<double> values, SampleBuilder sampleBuilder)
        {
            if (values        == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.values);
            if (sampleBuilder == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.sampleBuilder);

            return sampleBuilder.AddWithYield(values);
        }
    }
}
