using System.Collections;
using System.Collections.Generic;
using gfoidl.Stochastics.Enumerators;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample : IReadOnlyList<double>
    {
        /// <summary>
        /// Gets the value at the specified index in the <see cref="Sample" />.
        /// </summary>
        /// <param name="index">The zero-based index of the element to get.</param>
        /// <returns>The value at the specified index in the <see cref="Sample" />.</returns>
        /// <exception cref="System.ArgumentOutOfRangeException">
        /// <paramref name="index" /> is outside the bounds of <see cref="Sample" />.
        /// </exception>
        public double this[int index]
        {
            get
            {
                double[] values = _values;

                if ((uint)index >= (uint)values.Length)
                    ThrowHelper.ThrowArgumentOutOfRange(ThrowHelper.ExceptionArgument.index);

                return values[index];
            }
        }
        //---------------------------------------------------------------------
#pragma warning disable CS1591
        public ArrayEnumerator<double> GetEnumerator()          => new ArrayEnumerator<double>(_values);
        IEnumerator<double> IEnumerable<double>.GetEnumerator() => this.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator()                 => this.GetEnumerator();
#pragma warning restore CS1591
    }
}