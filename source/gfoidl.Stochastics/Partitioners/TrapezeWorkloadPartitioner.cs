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
     *          
     *  (0, A) is the load at the first iteration
     *  (N, B) is the load at the last iteration
     *  
     *  When there is no constant part of the load (say A=0 or B=0), so the partitioning can be 
     *  done as a triangle, thus the calculations are (greatly) simplified.
     *  
     *  Equation for trapeze: https://latex.codecogs.com/gif.latex?%5Clambda%20%3A%3D%20%5Cfrac%7BB-A%7D%7BN%7D%20%5Cqquad%20%5CRightarrow%20%5Cquad%20X%20%3D%20A%20&plus;%20%5Clambda%20%5Ccdot%20n%20%5C%5C%20%5C%5C%20A_p%20%3D%20%5Cfrac%7BA&plus;X%7D%7B2%7D%20%5Ccdot%20n%20%5Cqquad%20%5CLeftrightarrow%20%5Cquad%202A_p%20%3D%20%28A&plus;X%29%20%5Ccdot%20n%20%3D%20%282A%20&plus;%20%5Clambda%20%5Ccdot%20n%29%20%5Ccdot%20n%20%3D%202An%20&plus;%20%5Clambda%20%5Ccdot%20n%5E2%20%5C%5C%20%5C%5C%20%5Clambda%20%5Ccdot%20n%5E2%20&plus;%202A%20n%20-%202A_p%20%3D%200%20%5C%5C%20%5C%5C%20n%5E2%20&plus;%20%5Cfrac%7B2A%7D%7B%5Clambda%7D%20%5Ccdot%20n%20-%20%5Cfrac%7B2A_p%7D%7B%5Clambda%7D%20%3D%200%20%5Cqquad%20%5Cbigg%20%5Crvert%20%5Cquad%20p%3A%3D%20%5Cfrac%7B2A%7D%7B%5Clambda%7D%20%5Cquad%20%3B%20%5Cquad%20q%3A%3D-%5Cfrac%7B2A_p%7D%7B%5Clambda%7D%20%5C%5C%20%5C%5C%20n%20%3D%20-%5Cfrac%7Bp%7D%7B2%7D%20%5Cpm%20%5Csqrt%7B%5Cleft%28%20%5Cfrac%7Bp%7D%7B2%7D%20%5Cright%20%29%5E2%20-%20q%7D%20%5Cqquad%20%5CRightarrow%20%5Cboxed%7B%20n%20%28A_p%29%20%3D%20-%5Cfrac%7BA%7D%7B%5Clambda%7D%20%5Cpm%20%5Csqrt%7B%5Cleft%28%20%5Cfrac%7BA%7D%7B%5Clambda%7D%20%5Cright%20%29%5E2%20&plus;%20%5Cfrac%7B2%7D%7B%5Clambda%7D%20%5Ccdot%20A_p%7D%7D
     */
    internal sealed class TrapezeWorkloadPartitioner : WorkloadPartitioner
    {
        private readonly double _loadFactorAtStart;
        private readonly double _lambda;
        //---------------------------------------------------------------------
        public TrapezeWorkloadPartitioner(
            int    size,
            double loadFactorAtStart,           // A in schema above
            double loadFactorAtEnd,             // B in schema above
            int?   partitionCount = null)
            : base(size, partitionCount)
        {
            _loadFactorAtStart = loadFactorAtStart;
            _lambda            = (loadFactorAtEnd - loadFactorAtStart) / size;
        }
        //---------------------------------------------------------------------
        protected sealed override IEnumerable<KeyValuePair<long, Range>> GetOrderableDynamicPartitions(int partitionCount)
            => new TrapezeWorkloadPartitions(_size, _loadFactorAtStart, _lambda, partitionCount);
        //---------------------------------------------------------------------
        private class TrapezeWorkloadPartitions : Partitions
        {
            private readonly Range[] _partitions;
            //-----------------------------------------------------------------
            public TrapezeWorkloadPartitions(
                int    size,
                double loadFactorAtStart,
                double lambda,
                int    partitionCount)
                : base(size, partitionCount)
            {
                _partitions = lambda == 0
                    ? Rectangle()
                    : Trapeze();
                //-------------------------------------------------------------
                Range[] Rectangle()
                {
                    var partitions  = new Range[partitionCount];
                    var partitioner = new StaticRangePartitioner.StaticRangePartitions(size, partitionCount);

                    for (int i = 0; i < partitions.Length; ++i)
                        partitions[i] = partitioner.CalculateRange(i);

                    return partitions;
                }
                //-------------------------------------------------------------
                Range[] Trapeze()
                {
                    var partitions       = new Range[partitionCount];
                    double trapezeArea   = 0.5 * (2 * loadFactorAtStart + lambda * size) * size;
                    double partitionArea = trapezeArea / partitionCount;
                    double a_by_lambda   = loadFactorAtStart / lambda;
                    int start            = 0;

                    for (int i = 0; i < partitions.Length; ++i)
                    {
                        int end;

                        if (i == partitions.Length - 1)
                            end = _size;
                        else
                        {
                            double area = partitionArea * (i + 1);
                            double tmp  = Sqrt(a_by_lambda * a_by_lambda + 2d / lambda * area);

                            // +- in formula
                            if (lambda > 0)
                                tmp = -a_by_lambda + tmp;
                            else
                                tmp = -a_by_lambda - tmp;

                            end = (int)Round(tmp);
                        }

                        partitions[i] = (start, end);
                        start         = end;
                    }

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