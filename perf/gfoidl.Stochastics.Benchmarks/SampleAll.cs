using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Benchmarks
{
    public class SampleAll : IBenchmark
    {
        public void Run()
        {
#if !DEBUG
            BenchmarkRunner.Run<SampleAll>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(1_000, 1_000_000)]
        public int N { get; set; } = 100;
        //---------------------------------------------------------------------
        private double[] _values;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            var values = new double[this.N];
            var rnd = new Random(0);

            for (int i = 0; i < this.N; ++i)
                values[i] = rnd.NextDouble();

            _values = values;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double avg, double var) AverageAndVariance()
        {
            var sample = new Sample(_values);
            return (sample.Mean, sample.Variance);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double Delta()
        {
            var sample = new Sample(_values);
            return sample.Delta;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double min, double max) MinMax()
        {
            var sample = new Sample(_values);
            return (sample.Min, sample.Max);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double skewness, double kurtosis) SkewnessKurtosis()
        {
            var sample = new Sample(_values);
            return (sample.Skewness, sample.Kurtosis);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] ZTransformation()
        {
            var sample = new Sample(_values);
            return sample.ZTransformationToArray();
        }
    }
}
