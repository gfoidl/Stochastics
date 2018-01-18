using System;
using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Wrappers;

namespace gfoidl.Stochastics.Benchmarks
{
    //[DisassemblyDiagnoser(printSource: true)]
    [MemoryDiagnoser]
    public class ListWrapperBenchmarks
    {
        public static void Run()
        {
            var benchs      = new ListWrapperBenchmarks();
            benchs.N        = 1234;
            const int align = -25;
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.IList),align}: {benchs.IList()}");
            Console.WriteLine($"{nameof(benchs.List),align}: {benchs.List()}");
            Console.WriteLine($"{nameof(benchs.Wrapper),align}: {benchs.Wrapper()}");
#if !DEBUG
            BenchmarkRunner.Run<ListWrapperBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(100, 1_000, 10_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private List<double> _values;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            var rnd = new Random(0);

            _values = Enumerable
                .Repeat(0, this.N)
                .Select(_ => rnd.NextDouble())
                .ToList();
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double IList() => this.IList(_values);
        //---------------------------------------------------------------------
        [Benchmark]
        public double List() => this.List(_values);
        //---------------------------------------------------------------------
        [Benchmark]
        public double Wrapper() => this.Wrapper(new ListWrapper(_values));
        //---------------------------------------------------------------------
        private double IList(IList<double> source)
        {
            double sum = 0;
            int n      = source.Count;

            for (int i = 0; i < n; ++i)
                sum += source[i];

            return sum;
        }
        //---------------------------------------------------------------------
        private double List(List<double> source)
        {
            double sum = 0;
            int n      = source.Count;

            for (int i = 0; i < n; ++i)
                sum += source[i];

            return sum;
        }
        //---------------------------------------------------------------------
        private double Wrapper<TList>(TList source) where TList : IList<double>
        {
            double sum = 0;
            int n      = source.Count;

            for (int i = 0; i < n; ++i)
                sum += source[i];

            return sum;
        }
    }
}