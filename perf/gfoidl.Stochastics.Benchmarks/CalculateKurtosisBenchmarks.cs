using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Benchmarks
{
    public class CalculateKurtosisBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs      = new CalculateKurtosisBenchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.Simd),align}: {benchs.Simd()}");
            Console.WriteLine($"{nameof(benchs.ParallelizedSimd),align}: {benchs.ParallelizedSimd()}");
#if !DEBUG
            BenchmarkRunner.Run<CalculateKurtosisBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(1_000_000, 1_500_000, 2_000_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private Sample _sample;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            var values = new double[this.N];
            var rnd    = new Random(0);

            for (int i = 0; i < this.N; ++i)
                values[i] = rnd.NextDouble();

            _sample = new Sample(values);
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double Simd()
        {
            _sample.CalculateSkewnessAndKurtosisSimd(out double skewness, out double kurtosis);

            return skewness;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double ParallelizedSimd()
        {
            _sample.CalculateSkewnessAndKurtosisParallelizedSimd(out double skewness, out double kurtosis);

            return skewness;
        }
    }
}
