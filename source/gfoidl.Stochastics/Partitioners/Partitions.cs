using System.Collections;
using System.Collections.Generic;
using System.Threading;

namespace gfoidl.Stochastics.Partitioners
{
    internal abstract class Partitions : IEnumerable<KeyValuePair<long, Range>>
    {
#if NET_FULL
        private static int _instanceCounter;
        private int _instanceId;
#endif
        protected readonly int _size;
        protected readonly int _partitionCount;
        protected int          _partitionIndex = -1;
        //---------------------------------------------------------------------
        protected Partitions(int size, int partitionCount)
        {
            _partitionCount = partitionCount;
            _size           = size;
#if NET_FULL
            _instanceId = _instanceCounter++;
#endif
        }
        //---------------------------------------------------------------------
        IEnumerator IEnumerable.GetEnumerator() => this.GetEnumerator();
        //-----------------------------------------------------------------
        public IEnumerator<KeyValuePair<long, Range>> GetEnumerator()
        {
            while (true)
            {
                int partitionIndex = Interlocked.Increment(ref _partitionIndex);
#if NET_FULL
                Microsoft.ConcurrencyVisualizer.Instrumentation.Markers.WriteFlag(
                    "Range-Enumerator MoveNext, T-ID: {0}, Partitioner-ID: {1}, partitionIndex: {2}",
                    System.Threading.Thread.CurrentThread.ManagedThreadId,
                    _instanceId,
                    partitionIndex);
#endif

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