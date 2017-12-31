using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace gfoidl.Stochastics.Partitioners
{
    internal abstract class WorkloadPartitioner : OrderablePartitioner<Range>
    {
        protected readonly int _size;
        protected readonly int _partitionCount;
        //---------------------------------------------------------------------
        protected WorkloadPartitioner(int size, int? partitionCount = null)
            : base(true, true, true)
        {
            _size           = size;
            _partitionCount = partitionCount ?? Environment.ProcessorCount;
        }
        //---------------------------------------------------------------------
        public override bool SupportsDynamicPartitions => true;
        //---------------------------------------------------------------------
        // For Parallel.ForEach this method is not needed, only for PLinq
        public override IList<IEnumerator<KeyValuePair<long, Range>>> GetOrderablePartitions(int partitionCount)
        {
            var dynamicPartitions = this.GetOrderableDynamicPartitions(partitionCount);
            var partitions        = new IEnumerator<KeyValuePair<long, Range>>[partitionCount];

            for (int i = 0; i < partitions.Length; ++i)
                partitions[i] = dynamicPartitions.GetEnumerator();

            return partitions;
        }
        //---------------------------------------------------------------------
        public override IEnumerable<KeyValuePair<long, Range>> GetOrderableDynamicPartitions()
            => this.GetOrderableDynamicPartitions(_partitionCount);
        //---------------------------------------------------------------------
        protected abstract IEnumerable<KeyValuePair<long, Range>> GetOrderableDynamicPartitions(int partitionCount);
        //---------------------------------------------------------------------
        public static WorkloadPartitioner Create(int size, int? partitionCount = null)
            => new StaticRangePartitioner(size, partitionCount);
        //---------------------------------------------------------------------
        public static WorkloadPartitioner Create(int size, double lambda, int? partitionCount = null)
        {
            if (lambda == 0) return new StaticRangePartitioner(size, partitionCount);

            return new TrapezeWorkloadPartitioner(size, lambda, partitionCount);
        }
        //---------------------------------------------------------------------
        public static WorkloadPartitioner Create(
            int size,
            double loadFactorAtStart,
            double loadFactorAtEnd,
            int? partitionCount = null)
        {
            if (loadFactorAtStart == loadFactorAtEnd) return new StaticRangePartitioner(size, partitionCount);

            return new TrapezeWorkloadPartitioner(size, loadFactorAtStart, loadFactorAtEnd, partitionCount);
        }
    }
}