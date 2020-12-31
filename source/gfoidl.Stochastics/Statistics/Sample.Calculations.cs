using System;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private static ParallelOptions GetParallelOptions()
            => new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        //---------------------------------------------------------------------
        private double CalculateMedian()
        {
            _     = this.SortedValues;  // lazy created
            int n = _length;

            if ((n & 1) == 0)   // n % 2
            {
                return (_sortedValues[(n >> 1) - 1] + _sortedValues[n >> 1]) * 0.5;
            }
            else
            {
                // this is correct, but n is an int, so the next line is
                // very slight optimization and yield the same result.
                //_median = _sortedValues[(n - 1) / 2];
                return _sortedValues[n >> 1];
            }
        }
        //---------------------------------------------------------------------
        private double CalculateVariance()       => this.VarianceCore / this.Count;
        private double CalculateSampleVariance() => this.VarianceCore / (this.Count - 1d);
        //---------------------------------------------------------------------
        private double _varianceCore = double.NaN;
        private double VarianceCore
        {
            get
            {
                if (double.IsNaN(_varianceCore))
                    this.CalculateAverageAndVarianceCore();

                return _varianceCore;
            }
        }
    }
}
