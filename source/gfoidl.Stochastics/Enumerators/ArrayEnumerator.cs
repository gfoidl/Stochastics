﻿using System.Collections;
using System.Collections.Generic;

namespace gfoidl.Stochastics.Enumerators
{
#pragma warning disable CS1591
    public struct ArrayEnumerable<T> : IEnumerable<T>
    {
        private readonly T[] _array;
        //---------------------------------------------------------------------
        public ArrayEnumerable(T[] array)
        {
            if (array == null) ThrowHelper.ThrowArgumentNull(nameof(array));

            _array = array;
        }
        //---------------------------------------------------------------------
        public ArrayEnumerator<T> GetEnumerator()     => new ArrayEnumerator<T>(_array);
        IEnumerator<T> IEnumerable<T>.GetEnumerator() => this.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator()       => this.GetEnumerator();
    }
    //---------------------------------------------------------------------
    public struct ArrayEnumerator<T> : IEnumerator<T>
    {
        private readonly T[] _array;
        private int          _index;
        //---------------------------------------------------------------------
        public ArrayEnumerator(T[] array)
        {
            if (array == null) ThrowHelper.ThrowArgumentNull(nameof(array));

            _array = array;
            _index = -1;
        }
        //---------------------------------------------------------------------
        public T Current           => _array[_index];
        object IEnumerator.Current => this.Current;
        //---------------------------------------------------------------------
        public void Dispose() { }
        //---------------------------------------------------------------------
        public bool MoveNext()
        {
            _index++;
            return _index < _array.Length;
        }
        //-----------------------------------------------------------------
        public void Reset() => _index = -1;
    }
#pragma warning restore CS1591
}