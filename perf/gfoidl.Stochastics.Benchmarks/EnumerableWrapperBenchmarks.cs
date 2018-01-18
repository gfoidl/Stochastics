using System;
using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Wrappers;

namespace gfoidl.Stochastics.Benchmarks
{
    //[DisassemblyDiagnoser(printSource: true)]
    [MemoryDiagnoser]
    public class EnumerableWrapperBenchmarks
    {
        public static void Run()
        {
            var benchs      = new EnumerableWrapperBenchmarks();
            benchs.N        = 1234;
            const int align = -25;
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Base),align}: {benchs.Base()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Wrapper),align}: {benchs.Wrapper()}");
#if !DEBUG
            BenchmarkRunner.Run<EnumerableWrapperBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(100, 1_000, 10_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private IEnumerable<double> _values;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            _values = this.ProduceValues();
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double Base() => this.Base(_values);
        //---------------------------------------------------------------------
        [Benchmark]
        public double Wrapper() => this.Wrapper(EnumerableWrapper.Create(_values));
        //---------------------------------------------------------------------
        private double Base(IEnumerable<double> source)
        {
            double sum = 0;

            foreach (double item in source)
                sum += item;

            return sum;
        }
        //---------------------------------------------------------------------
        private double Wrapper<TEnumerable>(TEnumerable source) where TEnumerable : IMyIEnumerable
        {
            double sum = 0;

            foreach (double item in source)
                sum += item;

            return sum;
        }
        //---------------------------------------------------------------------
        private IEnumerable<double> ProduceValues()
        {
            var rnd = new Random(0);

            for (int i = 0; i < this.N; ++i)
                yield return rnd.NextDouble();
        }
    }
}