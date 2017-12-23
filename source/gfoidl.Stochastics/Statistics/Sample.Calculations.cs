namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
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
        private double CalculateVariance()       => this.VarianceCore() / this.Count;
        private double CalculateSampleVariance() => this.VarianceCore() / (this.Count - 1d);
        //---------------------------------------------------------------------
        private double _varianceCore = double.NaN;
        private double VarianceCore()
        {
            if (double.IsNaN(_varianceCore))
                _varianceCore = this.CalculateVarianceCore();

            return _varianceCore;
        }
        //---------------------------------------------------------------------
        private double CalculateKurtosis()
        {
            double kurtosis = 0;
            double[] tmp    = _values;
            double avg      = this.Mean;
            double sigma    = this.StandardDeviation;

            for (int i = 0; i < tmp.Length; ++i)
            {
                double t = tmp[i] - avg;
                kurtosis += t * t * t * t;
            }

            kurtosis /= tmp.Length * sigma * sigma * sigma * sigma;

            return kurtosis;
        }
        //---------------------------------------------------------------------
        private double ZTransformation(double value, double avg, double sigma) => (value - avg) / sigma;
    }
}