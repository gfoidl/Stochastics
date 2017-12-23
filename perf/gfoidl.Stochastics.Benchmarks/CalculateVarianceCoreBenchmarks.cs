//#define SEQ
//-----------------------------------------------------------------------------
using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    //[DisassemblyDiagnoser(printSource: true)]
    public class CalculateVarianceCoreBenchmarks
    {
        public static void Run()
        {
            var benchs = new CalculateVarianceCoreBenchmarks();
            benchs.N = 1000;
            benchs.GlobalSetup();
            Console.WriteLine(benchs.Sequential());
            Console.WriteLine(benchs.Simd());
            //Console.WriteLine(benchs.Parallelized());
            Console.WriteLine(benchs.ParallelizedSimd());
#if !DEBUG
            BenchmarkRunner.Run<CalculateVarianceCoreBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(100, 1_000, 10_000, 100_000, 1_000_000)]
        public int N { get; set; }
        //---------------------------------------------------------------------
        private double[] _values;
        private double   _avg;
        //---------------------------------------------------------------------
        public double Mean => _avg;
        public int Count   => _values.Length;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            _values = new double[this.N];
            _avg    = 0;
            var rnd = new Random(0);

            for (int i = 0; i < this.N; ++i)
            {
                _values[i] = rnd.NextDouble();
                _avg += _values[i];
            }

            _avg /= this.N;
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double Sequential()
        {
            double variance = 0;
            double[] arr    = _values;
            double avg      = this.Mean;

            for (int i = 0; i < arr.Length; ++i)
                variance += arr[i] * arr[i];

            variance -= arr.Length * avg * avg;

            return variance;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double Simd()
        {
            double variance = 0;
            double[] arr    = _values;
            double avg      = this.Mean;
            int i           = 0;

            if (Vector.IsHardwareAccelerated && this.Count >= Vector<double>.Count * 2)
            {
                for (; i < arr.Length - 2 * Vector<double>.Count; i += Vector<double>.Count)
                {
                    var v1 = new Vector<double>(arr, i);
                    var v2 = new Vector<double>(arr, i);
                    variance += Vector.Dot(v1, v2);

                    i += Vector<double>.Count;
                    v1 = new Vector<double>(arr, i);
                    v2 = new Vector<double>(arr, i);
                    variance += Vector.Dot(v1, v2);
                }
            }

            for (; i < arr.Length; ++i)
                variance += arr[i] * arr[i];

            variance -= arr.Length * avg * avg;

            return variance;
        }
        //---------------------------------------------------------------------
        //[Benchmark]
        public double Parallelized()
        {
            double variance = 0;
            var sync        = new object();

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);
            Parallel.ForEach(
                partitioner,
                range =>
                {
                    double local = 0;
                    double[] arr = _values;
                    double avg   = this.Mean;
                    int i        = range.Item1;

                    // RCE
                    if ((uint)range.Item1 >= arr.Length || (uint)range.Item2 > arr.Length)
                        ThrowHelper.ThrowArgumentOutOfRange(nameof(range));

                    for (; i < range.Item2; ++i)
                        local += arr[i] * arr[i];

                    lock (sync) variance += local;
                }
            );

            variance -= this.Count * this.Mean * this.Mean;
            return variance;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double ParallelizedSimd()
        {
            double variance = 0;
            var sync        = new object();

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);
#if SEQ
            var range = Tuple.Create(0, _values.Length);
#else
            Parallel.ForEach(
                partitioner,
                range =>
                {
#endif
                    double local = 0;
                    double[] arr = _values;
                    double avg   = this.Mean;
                    int i        = range.Item1;

                    // RCE
                    if ((uint)range.Item1 >= arr.Length || (uint)range.Item2 > arr.Length)
                        ThrowHelper.ThrowArgumentOutOfRange(nameof(range));

                    if (Vector.IsHardwareAccelerated && (range.Item2 - range.Item1) >= Vector<double>.Count * 2)
                    {
                        for (; i < range.Item2 - 2 * Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var v1 = new Vector<double>(arr, i);
                            var v2 = new Vector<double>(arr, i);
                            local += Vector.Dot(v1, v2);

                            i += Vector<double>.Count;
                            v1 = new Vector<double>(arr, i);
                            v2 = new Vector<double>(arr, i);
                            local += Vector.Dot(v1, v2);
                        }
                    }

                    for (; i < range.Item2; ++i)
                        local += arr[i] * arr[i];

                    lock (sync) variance += local;
#if !SEQ
                }
            );
#endif
            variance -= this.Count * this.Mean * this.Mean;
            return variance;
        }
    }
}