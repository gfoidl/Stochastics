namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// Configuration for the thresholds used in the calculations of <see cref="Sample" />.
    /// </summary>
    /// <remarks>
    /// Threshould determined by benchmark (roughly).
    /// </remarks>
    public static class SampleThresholds
    {
        /// <summary>
        /// Threshold for sequential vs. parallel execution.
        /// </summary>
        public static int ThresholdForParallel { get; set; } = 1_500_000;

        /// <summary>
        ///  Threshold for sequential vs. parallel execution for <see cref="Sample.AutoCorrelation" />.
        /// </summary>
        public static int ThresholdForAutocorrelationParallel { get; set; } = 1_500;

        /// <summary>
        ///  Threshold for sequential vs. parallel execution for <see cref="Sample.Min" />
        ///  and <see cref="Sample.Max" />.
        /// </summary>
        public static int ThresholdForMinMax { get; set; } = 1_750_000;
    }
}
