using System.Collections;
using System.Collections.Generic;

namespace gfoidl.Stochastics.Enumerators
{
#pragma warning disable CS1591
    public struct ArrayEnumerable<T> : IEnumerable<T>
    {
        private readonly T[] _array;
        private readonly int _offset;
        private readonly int _length;
        //---------------------------------------------------------------------
        public ArrayEnumerable(T[] array) : this(array, 0, array.Length) { }
        //---------------------------------------------------------------------
        public ArrayEnumerable(T[] array, int offset, int length)
        {
            if (array == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.array);

            _array  = array;
            _offset = offset;
            _length = length;
        }
        //---------------------------------------------------------------------
        public ArrayEnumerator<T> GetEnumerator()     => new ArrayEnumerator<T>(_array, _offset, _length);
        IEnumerator<T> IEnumerable<T>.GetEnumerator() => this.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator()       => this.GetEnumerator();
        //---------------------------------------------------------------------
        public static implicit operator ArrayEnumerable<T>(T[] array)       => new ArrayEnumerable<T>(array);
        public static implicit operator T[] (ArrayEnumerable<T> enumerable) => enumerable._array;
    }
    //---------------------------------------------------------------------
    public struct ArrayEnumerator<T> : IEnumerator<T>
    {
        private readonly T[] _array;
        private readonly int _offset;
        private readonly int _length;
        private int          _index;
        //---------------------------------------------------------------------
        public ArrayEnumerator(T[] array) : this(array, 0, array.Length) { }
        //---------------------------------------------------------------------
        public ArrayEnumerator(T[] array, int offset, int length)
        {
            if (array == null) ThrowHelper.ThrowArgumentNull(ThrowHelper.ExceptionArgument.array);

            _array  = array;
            _offset = offset;
            _length = length;
            _index  = -1;
        }
        //---------------------------------------------------------------------
        public T Current           => _array[_offset + _index];
        object IEnumerator.Current => this.Current;
        //---------------------------------------------------------------------
        public void Dispose() { }
        //---------------------------------------------------------------------
        public bool MoveNext()
        {
            _index++;
            return (uint)_index < (uint)(_length);
        }
        //-----------------------------------------------------------------
        public void Reset() => _index = -1;
    }
#pragma warning restore CS1591
}
