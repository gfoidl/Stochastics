using System;
using System.Text;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Benchmarks
{
    public class AutoCorrelationBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs      = new AutoCorrelationBenchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.Simd),align}: {GetString(benchs.Simd())}");
            Console.WriteLine($"{nameof(benchs.ParallelizedSimd),align}: {GetString(benchs.ParallelizedSimd())}");
#if !DEBUG
            BenchmarkRunner.Run<AutoCorrelationBenchmarks>();
#endif
            //-----------------------------------------------------------------
            string GetString(double[] array)
            {
                var sb = new StringBuilder();
                int n  = Math.Min(10, array.Length);

                for (int i = 0; i < n; ++i)
                {
                    sb.Append(array[i]);

                    if (i < n - 1) sb.Append(", ");
                }
                sb.Append("  N: ").Append(array.Length);

                return sb.ToString();
            }
        }
        //---------------------------------------------------------------------
        [Params(10_000, 50_000)]
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
        public double[] Simd() => _sample.AutoCorrelationToArraySimd();
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] ParallelizedSimd() => _sample.AutoCorrelationToArrayParallelSimd();
    }
}
