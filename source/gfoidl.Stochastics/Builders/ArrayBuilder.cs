﻿using System;
using System.Collections.Generic;

/*
 * Inspired from https://github.com/dotnet/corefx/blob/master/src/Common/src/System/Collections/Generic/LargeArrayBuilder.cs
 */

namespace gfoidl.Stochastics.Builders
{
    internal struct ArrayBuilder<T>
    {
        private const int StartCapacity   = 8;
        private const int ResizeThreshold = 16;
        private readonly int _maxCapacity;
        private T[]          _firstBuffer;
        private List<T[]>    _buffers;
        private T[]          _currentBuffer;
        private int          _index;
        private int          _count;
        //---------------------------------------------------------------------
        public ArrayBuilder(bool initialize) : this(int.MaxValue) { }
        //---------------------------------------------------------------------
        public ArrayBuilder(int maxCapacity) : this()
        {
            _maxCapacity = maxCapacity;
            _firstBuffer = _currentBuffer = new T[StartCapacity];
            _buffers     = new List<T[]>();
        }
        //---------------------------------------------------------------------
        public void Add(T item)
        {
            if (_index == _currentBuffer.Length)
                this.AllocateBuffer();

            _currentBuffer[_index++] = item;
            _count++;
        }
        //---------------------------------------------------------------------
        public T[] ToArray()
        {
            if (this.TryMove(out T[] array))
                return array;

            array = new T[_count];
            this.CopyTo(array);

            return array;
        }
        //---------------------------------------------------------------------
        private bool TryMove(out T[] array)
        {
            array = _firstBuffer;

            return _count == _firstBuffer.Length;
        }
        //---------------------------------------------------------------------
        private void CopyTo(T[] array)
        {
            int arrayIndex = 0;
            int count      = _count;

            for (int i = 0; count > 0; ++i)
            {
                T[] buffer = this.GetBuffer(i);
                int toCopy = Math.Min(count, buffer.Length);
                Array.Copy(buffer, 0, array, arrayIndex, toCopy);

                count      -= toCopy;
                arrayIndex += toCopy;
            }
        }
        //---------------------------------------------------------------------
        private void AllocateBuffer()
        {
            if ((uint)_count < (uint)ResizeThreshold)
            {
                int newCapacity = Math.Min(_count == 0 ? StartCapacity : _count * 2, _maxCapacity);
                _currentBuffer  = new T[newCapacity];
                Array.Copy(_firstBuffer, 0, _currentBuffer, 0, _count);
                _firstBuffer    = _currentBuffer;
            }
            else
            {
                int newCapacity = ResizeThreshold;

                if (_count != ResizeThreshold)
                {
                    // Example scenario: Let's say _count == 64.
                    // Then our buffers look like this: | 8 | 8 | 16 | 32 |
                    // As you can see, our count will be just double the last buffer.
                    // Now, say _maxCapacity is 100. We will find the right amount to allocate by
                    // doing min(64, 100 - 64). The lhs represents double the last buffer,
                    // the rhs the limit minus the amount we've already allocated.

                    _buffers.Add(_currentBuffer);
                    newCapacity = Math.Min(_count, _maxCapacity - _count);
                }

                _currentBuffer = new T[newCapacity];
                _index = 0;
            }
        }
        //---------------------------------------------------------------------
        private T[] GetBuffer(int index)
        {
            return index == 0
                ? _firstBuffer
                : index <= _buffers.Count
                    ? _buffers[index - 1]       // first "buffer" is _firstBuffer resized
                    : _currentBuffer;
        }
    }
}