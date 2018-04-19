using System.Collections.Generic;

namespace gfoidl.Stochastics.Partitioners
{
    internal sealed class StaticRangePartitioner : WorkloadPartitioner
    {
        public StaticRangePartitioner(int size, int? partitionCount = null)
            : base(size, partitionCount)
        { }
        //---------------------------------------------------------------------
        protected sealed override IEnumerable<KeyValuePair<long, Range>> GetOrderableDynamicPartitions(int partitionCount)
            => new StaticRangePartitions(_size, partitionCount);
        //---------------------------------------------------------------------
        public class StaticRangePartitions : Partitions
        {
            private readonly int _partitionSize;
            //---------------------------------------------------------------------
            public StaticRangePartitions(int size, int partitionCount)
                : base(size, partitionCount) 
                => _partitionSize = size / partitionCount;
            //---------------------------------------------------------------------
            protected internal override Range CalculateRange(int partitionIndex)
            {
                int start = partitionIndex * _partitionSize;
                int end;

                if (partitionIndex < _partitionCount)
                    end = start + _partitionSize;
                else
                    end = _size;

                if (start == end) return Range.Null;

                return (start, end);
            }
        }
    }
}
