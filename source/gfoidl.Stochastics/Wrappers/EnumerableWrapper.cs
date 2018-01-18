using System.Collections;
using System.Collections.Generic;

namespace gfoidl.Stochastics.Wrappers
{
    internal interface IMyIEnumerable : IEnumerable<double>
    {
        new EnumerableWrapper.EnumerableIterator GetEnumerator();
    }
    //---------------------------------------------------------------------
    internal readonly struct EnumerableWrapper : IMyIEnumerable
    {
        private readonly IEnumerable<double> _enumerable;
        //---------------------------------------------------------------------
        public EnumerableWrapper(IEnumerable<double> enumerable)               => _enumerable = enumerable;
        public static EnumerableWrapper Create(IEnumerable<double> enumerable) => new EnumerableWrapper(enumerable);
        //---------------------------------------------------------------------
        IEnumerator<double> IEnumerable<double>.GetEnumerator() => this.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator()                 => this.GetEnumerator();
        public EnumerableIterator GetEnumerator()               => new EnumerableIterator(_enumerable);
        //---------------------------------------------------------------------
        public struct EnumerableIterator : IEnumerator<double>
        {
            private readonly IEnumerator<double> _enumerator;
            //---------------------------------------------------------------------
            public EnumerableIterator(IEnumerable<double> enumerable) => _enumerator = enumerable.GetEnumerator();
            //---------------------------------------------------------------------
            public double Current      => _enumerator.Current;
            object IEnumerator.Current => this.Current;
            public void Dispose()      => _enumerator.Dispose();
            public bool MoveNext()     => _enumerator.MoveNext();
            public void Reset()        => _enumerator.Reset();
        }
    }
}