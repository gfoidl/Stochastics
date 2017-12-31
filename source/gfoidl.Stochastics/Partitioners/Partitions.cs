using System.Collections;
using System.Collections.Generic;
using System.Threading;

namespace gfoidl.Stochastics.Partitioners
{
    internal abstract class Partitions : IEnumerable<KeyValuePair<long, Range>>
    {
        protected readonly int _size;
        protected readonly int _partitionCount;
        protected int          _partitionIndex;
        //---------------------------------------------------------------------
        protected Partitions(int size, int partitionCount)
        {
            _partitionCount = partitionCount;
            _size           = size;
        }
        //---------------------------------------------------------------------
        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
        //-----------------------------------------------------------------
        public IEnumerator<KeyValuePair<long, Range>> GetEnumerator()
        {
            while (true)
            {
                int partitionIndex = Interlocked.Increment(ref _partitionIndex) - 1;

                if (partitionIndex > _partitionCount) yield break;

                Range range = this.CalculateRange(partitionIndex);

                if (range == Range.Null) yield break;

                yield return new KeyValuePair<long, Range>(partitionIndex, range);
            }
        }
        //---------------------------------------------------------------------
        protected internal abstract Range CalculateRange(int partitionIndex);
    }
}