using System.Collections.Generic;
using static System.Math;

namespace gfoidl.Stochastics.Partitioners
{
    /*
     * A partitioner for trapeze shaped workloads.
     * Total work is the area of the trapeze. So partition in equal sized
     * areas.
     * 
     *              A
     *              |\
     *              |  \
     *              |   |\
     *              |   |  \ X
     *              |   |    \
     *              |   |    | \
     *              |   |    |   \ B
     *              |   |    |     \
     *              | A1| A2 | A3  |
     *              |   |    |     |
     *              |___|____|_____| 
     *              0        n     N
     *          
     *  (0, A) is the load at the first iteration
     *  (N, B) is the load at the last iteration
     *  
     *  The constant part with load to B can be eliminated, so the partitioning can be done on a triangle.
     *  Thus the calculations are (greatly) simplified.
     */
    internal class TrapezeWorkloadPartitioner : WorkloadPartitioner
    {
        private readonly double _lambda;
        //---------------------------------------------------------------------
        public TrapezeWorkloadPartitioner(
            int    size,
            double lambda,
            int?   partitionCount = null)
            : base(size, partitionCount)
            => _lambda = lambda;
        //---------------------------------------------------------------------
        public TrapezeWorkloadPartitioner(
            int    size,
            double loadFactorAtStart,
            double loadFactorAtEnd,
            int?   partitionCount = null)
            : this(size, (loadFactorAtEnd - loadFactorAtStart) / size, partitionCount)
        { }
        //---------------------------------------------------------------------
        protected override IEnumerable<KeyValuePair<long, Range>> GetOrderableDynamicPartitions(int partitionCount)
            => new TrapezeWorkloadPartitions(_size, _lambda, partitionCount);
        //---------------------------------------------------------------------
        private class TrapezeWorkloadPartitions : Partitions
        {
            private readonly Range[] _partitions;
            //-----------------------------------------------------------------
            public TrapezeWorkloadPartitions(
                int    size,
                double lambda,
                int    partitionCount)
                : base(size, partitionCount)
            {
                _partitions = lambda == 0
                    ? Rectangle()
                    : Triangle();
                //-------------------------------------------------------------
                Range[] Triangle()
                {
                    bool increasing = true;
                    if (lambda < 0)
                    {
                        increasing = false;
                        lambda     = -lambda;
                    }

                    var partitions       = new Range[partitionCount];
                    double triangleArea  = (double)size * size * lambda * 0.5;
                    double partitionArea = triangleArea / partitionCount;
                    double lambdaInv     = 1d / lambda;
                    int start            = 0;

                    for (int i = 0; i < partitions.Length; ++i)
                    {
                        int end;

                        if (i == partitions.Length - 1)
                            end = _size;
                        else
                        {
                            double area = partitionArea * (i + 1);
                            double tmp  = Sqrt(2 * area * lambdaInv);
                            end         = (int)Round(tmp);
                        }

                        if (increasing)
                            partitions[i] = (start, end);
                        else
                            partitions[partitionCount - 1 - i] = (size - end, size - start);

                        start = end;
                    }

                    return partitions;
                }
                //-------------------------------------------------------------
                Range[] Rectangle()
                {
                    var partitions  = new Range[partitionCount];
                    var partitioner = new StaticRangePartitioner.StaticRangePartitions(size, partitionCount);

                    for (int i = 0; i < partitions.Length; ++i)
                        partitions[i] = partitioner.CalculateRange(i);

                    return partitions;
                }
            }
            //-----------------------------------------------------------------
            protected internal override Range CalculateRange(int partitionIndex)
            {
                if ((uint)partitionIndex >= (uint)_partitions.Length)
                    return Range.Null;

                return _partitions[partitionIndex];
            }
        }
    }
}