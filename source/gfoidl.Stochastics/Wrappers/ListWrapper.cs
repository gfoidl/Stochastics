using System.Collections;
using System.Collections.Generic;

namespace gfoidl.Stochastics.Wrappers
{
    internal readonly struct ListWrapper : IList<double>
    {
        private readonly IList<double> _list;
        //---------------------------------------------------------------------
        public ListWrapper(IList<double> list) => _list = list;
        //---------------------------------------------------------------------
        public double this[int index]
        {
            get => _list[index];
            set => _list[index] = value;
        }
        //---------------------------------------------------------------------
        public int Count                                   => _list.Count;
        public bool IsReadOnly                             => false;
        public void Add(double item)                       => _list.Add(item);
        public void Clear()                                => _list.Clear();
        public bool Contains(double item)                  => _list.Contains(item);
        public void CopyTo(double[] array, int arrayIndex) => _list.CopyTo(array, arrayIndex);
        public int IndexOf(double item)                    => _list.IndexOf(item);
        public void Insert(int index, double item)         => _list.Insert(index, item);
        public bool Remove(double item)                    => _list.Remove(item);
        public void RemoveAt(int index)                    => _list.RemoveAt(index);
        //---------------------------------------------------------------------
        public ListIterator GetEnumerator()                     => new ListIterator(_list);
        IEnumerator<double> IEnumerable<double>.GetEnumerator() => this.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator()                 => this.GetEnumerator();
        //---------------------------------------------------------------------
        public struct ListIterator : IEnumerator<double>
        {
            private readonly IList<double> _list;
            private int                    _index;
            //---------------------------------------------------------------------
            public ListIterator(IList<double> list)
            {
                _list  = list;
                _index = -1;
            }
            //---------------------------------------------------------------------
            public double Current      => _list[_index];
            object IEnumerator.Current => this.Current;
            //---------------------------------------------------------------------
            public bool MoveNext()
            {
                _index++;

                return _index < _list.Count;
            }
            //---------------------------------------------------------------------
            public void Dispose() { }
            public void Reset() => _index = -1;
        }
    }
}