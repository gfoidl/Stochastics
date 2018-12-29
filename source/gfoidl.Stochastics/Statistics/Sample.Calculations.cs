using System;
using System.Threading.Tasks;
using gfoidl.Stochastics.Native;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        private static ParallelOptions GetParallelOptions()
            => new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        //---------------------------------------------------------------------
        /// <summary>
        /// Calculates the statistics for <see cref="Sample" />.
        /// </summary>
        /// <remarks>
        /// The statistical properties of <see cref="Sample" /> are lazy-evaluated.
        /// With this method these properties are instantly evalualted / calculated.
        /// </remarks>
        public void CalculateStats()
        {
            // is threadsafe, because from shared state is just read
            Task<double> medianTask = Task.Run(() => this.CalculateMedian());

            if (Gpu.IsAvailable && (Gpu.IsUseOfGpuForced || this.Count > SampleThresholds.ThresholdForGpu))
                Gpu.CalculateSampleStats(this);
            else
            {
                this.CalculateAverageAndVarianceCore();
                this.GetMinMax();
                this.CalculateDelta();
                this.CalculateSkewnessAndKurtosis();
            }

            medianTask.GetAwaiter().GetResult();
        }
        //---------------------------------------------------------------------
        private double CalculateMedian()
        {
            int n = this.SortedValues.Count;

            if ((n & (2 - 1)) == 0)     // n % 2 == 0
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
        internal double VarianceCore
        {
            get
            {
                if (double.IsNaN(_varianceCore))
                    this.CalculateAverageAndVarianceCore();

                return _varianceCore;
            }
            set => _varianceCore = value;
        }
    }
}
