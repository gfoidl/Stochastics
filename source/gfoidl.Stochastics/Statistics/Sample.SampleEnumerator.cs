using System;
using System.Collections;
using System.Collections.Generic;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample : IReadOnlyList<double>
    {
        public double this[int index] => _values[index];
        //---------------------------------------------------------------------
        public SampleEnumerator GetEnumerator()                 => new SampleEnumerator(_values);
        IEnumerator<double> IEnumerable<double>.GetEnumerator() => this.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator()                 => this.GetEnumerator();
        //---------------------------------------------------------------------
        public struct SampleEnumerator : IEnumerator<double>
        {
            private readonly double[] _values;
            private int               _index;
            //---------------------------------------------------------------------
            public SampleEnumerator(double[] values)
            {
                _values = values;
                _index  = -1;
            }
            //---------------------------------------------------------------------
            public double Current      => _values[_index];
            object IEnumerator.Current => this.Current;
            //---------------------------------------------------------------------
            public void Dispose() { }
            //---------------------------------------------------------------------
            public bool MoveNext()
            {
                _index++;
                return _index < _values.Length;
            }
            //-----------------------------------------------------------------
            public void Reset() => _index = -1;
        }
    }
}