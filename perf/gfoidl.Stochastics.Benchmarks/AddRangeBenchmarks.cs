using System;
using System.Collections.Generic;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class AddRangeBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs      = new AddRangeBenchmarks();
            benchs.N        = 100_000;
            benchs.GlobalSetup();
            const int align = -35;
            Console.WriteLine($"{nameof(benchs.Default),align}: {benchs.Default()}");
            Console.WriteLine($"{nameof(benchs.AddRange),align}: {benchs.AddRange()}");
            Console.WriteLine($"{nameof(benchs.AddRangeArray),align}: {benchs.AddRangeArray()}");
#if !DEBUG
            BenchmarkRunner.Run<AddRangeBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(100, 1_000, 10_000, 100_000, 1_000_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private double[]     _array;
        private List<double> _list;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            _array  = new double[this.N];
            _list   = new List<double>(this.N);
            var rnd = new Random(0);

            for (int i = 0; i < this.N; ++i)
            {
                double val = rnd.NextDouble();
                _array[i]  = val;
                _list.Add(val);
            }
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public (double min, double max, double avg) Default()
        {
            //var sample = new Sample(_list);
            var sample = new Sample(this.GetItems());

            return (sample.Min, sample.Max, sample.Mean);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double min, double max, double avg) AddRange()
        {
            var sample = new Sample();
            //sample.AddRange(_list);
            sample.AddRange(this.GetItems());

            return (sample.Min, sample.Max, sample.Mean);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double min, double max, double avg) AddRangeArray()
        {
            var sample = new Sample();
            sample.AddRange(_array);

            return (sample.Min, sample.Max, sample.Mean);
        }
        //---------------------------------------------------------------------
        private IEnumerable<double> GetItems()
        {
            var rnd = new Random(0);
            int n   = this.N;

            for (int i = 0; i < n; ++i)
                yield return rnd.NextDouble();
        }
    }
}