using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Benchmarks
{
    public class GetMinMaxBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs      = new GetMinMaxBenchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.Simd),align}: {benchs.Simd()}");
            Console.WriteLine($"{nameof(benchs.ParallelizedSimd),align}: {benchs.ParallelizedSimd()}");
#if !DEBUG
            BenchmarkRunner.Run<GetMinMaxBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(1_750_000, 2_000_000)]
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
        public (double Min, double Max) Simd()
        {
            _sample.GetMinMaxSimd(out double min, out double max);

            return (min, max);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Min, double Max) ParallelizedSimd()
        {
            _sample.GetMinMaxParallelizedSimd(out double min, out double max);

            return (min, max);
        }
    }
}
