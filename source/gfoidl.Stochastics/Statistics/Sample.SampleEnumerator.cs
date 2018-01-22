using System.Collections;
using System.Collections.Generic;
using gfoidl.Stochastics.Enumerators;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample : IReadOnlyList<double>
    {
        public double this[int index] => _values[index];
        //---------------------------------------------------------------------
        public ArrayEnumerator<double> GetEnumerator()          => new ArrayEnumerator<double>(_values);
        IEnumerator<double> IEnumerable<double>.GetEnumerator() => this.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator()                 => this.GetEnumerator();
    }
}