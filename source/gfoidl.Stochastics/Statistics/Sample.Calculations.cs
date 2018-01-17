using System;
using System.Threading.Tasks;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
#pragma warning disable CS1591
        // Threshould determined by benchmark (roughly)
        public const int ThresholdForParallel                = 50_000;
        public const int ThresholdForAutocorrelationParallel = 250;
        public const int ThresholdForMinMax                  = 75_000;
#pragma warning restore CS1591
        //---------------------------------------------------------------------
        private static ParallelOptions GetParallelOptions()
            => new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        //---------------------------------------------------------------------
        private double CalculateMedian()
        {
            int n = this.SortedValues.Count;

            if (n % 2 == 0)
                return (_sortedValues[(n >> 1) - 1] + _sortedValues[n >> 1]) * 0.5;
            else
                // this is correct, but n is an int, so the next line is 
                // very slight optimization and yield the same result.
                //_median = _sortedValues[(n - 1) / 2];
                return _sortedValues[n >> 1];
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