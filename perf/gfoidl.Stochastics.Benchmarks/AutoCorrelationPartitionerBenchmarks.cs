//#define STATS
//-----------------------------------------------------------------------------
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Partitioners;

#if NET_FULL
using Microsoft.ConcurrencyVisualizer.Instrumentation;
#endif

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class AutoCorrelationPartitionerBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs = new AutoCorrelationPartitionerBenchmarks();
#if DEBUG
            benchs.N = 2_500;
#else
            benchs.N = 350_000;
#endif
            Console.WriteLine(benchs.N);
            benchs.GlobalSetup();
            {
                benchs.PfxDefault();
                benchs.StaticRange();
                benchs.TrapezeWorkload();
                benchs.CustomLoop();
            }
            benchs.GlobalCleanUp();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.PfxDefault),align}: {GetString(benchs.PfxDefault())}");
            Console.WriteLine($"{nameof(benchs.StaticRange),align}: {GetString(benchs.StaticRange())}");
            Console.WriteLine($"{nameof(benchs.TrapezeWorkload),align}: {GetString(benchs.TrapezeWorkload())}");
            Console.WriteLine($"{nameof(benchs.CustomLoop),align}: {GetString(benchs.CustomLoop())}");
#if !DEBUG
            BenchmarkRunner.Run<AutoCorrelationPartitionerBenchmarks>();
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
#pragma warning disable CS0649
        private ConcurrentDictionary<int, HashSet<Range>> _ranges;
#pragma warning restore CS0649
        private string _name;
        //---------------------------------------------------------------------
        [Params(1_000, 10_000, 20_000)]
        public int N { get; set; } = 5_000;
        //---------------------------------------------------------------------
        [Params(1, 2, 3, 4)]
        public int PartitionMultiplier { get; set; } = 8;
        //---------------------------------------------------------------------
        private double[] _values;
        public int Count => _values.Length;
        //---------------------------------------------------------------------
        [Conditional("STATS")]
        private void AddRange(Range range)
        {
            var list = new HashSet<Range> { range };
            _ranges.AddOrUpdate(
                Thread.CurrentThread.ManagedThreadId,
                list,
                (id, l) =>
                {
                    l.Add(range);
                    return l;
                }
            );
        }
        //---------------------------------------------------------------------
        [Conditional("STATS")]
        private void ClearRanges() => _ranges.Clear();
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
#if STATS
            _ranges = new ConcurrentDictionary<int, HashSet<Range>>();
#endif
            _values = new double[this.N];
            var rnd = new Random(0);

            for (int i = 0; i < this.N; ++i)
                _values[i] = rnd.NextDouble();
        }
        //---------------------------------------------------------------------
        [GlobalCleanup, Conditional("STATS")]
        public void GlobalCleanUp()
        {
            string baseDir = Environment.GetEnvironmentVariable("BENCH_RES_DIR") ?? AppContext.BaseDirectory;
            //string baseDir = "/home/rsa-key-20171228/bench/perf/gfoidl.Stochastics.Benchmarks";
            string fileName = Path.Combine(baseDir, $"{_name}_{Environment.TickCount}.txt");

            using (StreamWriter sw = File.CreateText(fileName))
            {
                foreach (var item in _ranges.OrderBy(i => i.Key))
                {
                    sw.WriteLine($"Thread-ID: {item.Key}");

                    foreach (var range in item.Value.OrderBy(r => r.Start))
                        sw.WriteLine($"\t{range,-15}");
                }
            }
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double[] PfxDefault()
        {
            _name = nameof(PfxDefault);
            this.ClearRanges();

            var corr = new double[_values.Length / 2];

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length / 2),
                range => this.AutoCorrelationToArrayImpl(corr, (range.Item1, range.Item2))
            );

            return corr;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] StaticRange()
        {
            _name = nameof(StaticRange);
            this.ClearRanges();

            var corr = new double[_values.Length / 2];

            Parallel.ForEach(
                WorkloadPartitioner.Create(_values.Length / 2),
                range => this.AutoCorrelationToArrayImpl(corr, range)
            );

            return corr;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] TrapezeWorkload()
        {
            _name = nameof(TrapezeWorkload);
            this.ClearRanges();

            int n    = _values.Length;
            var corr = new double[n / 2];

            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 8 };

            Parallel.ForEach(
                WorkloadPartitioner.Create(n / 2, loadFactorAtStart: n, loadFactorAtEnd: n * 0.5, Environment.ProcessorCount * this.PartitionMultiplier),
                parallelOptions,
                range => this.AutoCorrelationToArrayImpl(corr, range));

            return corr;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] CustomLoop()
        {
            _name = nameof(CustomLoop);
            this.ClearRanges();

            int n    = _values.Length;
            var corr = new double[n / 2];

            int parallelCount               = Environment.ProcessorCount;
            WorkloadPartitioner partitioner = WorkloadPartitioner.Create(n / 2, loadFactorAtStart: n, loadFactorAtEnd: n * 0.5, Environment.ProcessorCount * this.PartitionMultiplier);
            IEnumerable<KeyValuePair<long, Range>> partitionEnumerator = partitioner.GetOrderableDynamicPartitions();

            var tasks = new Task[parallelCount];
            tasks[0]  = new Task(Loop, partitionEnumerator);

            for (int i = 1; i < parallelCount; ++i)
                tasks[i] = Task.Factory.StartNew(Loop, partitionEnumerator);

            tasks[0].RunSynchronously();
            Task.WaitAll(tasks);

            return corr;
            //-----------------------------------------------------------------
            void Loop(object _)
            {
                foreach (KeyValuePair<long, Range> partition in _ as IEnumerable<KeyValuePair<long, Range>>)
                {
                    Range range = partition.Value;
                    this.AutoCorrelationToArrayImpl(corr, range);
                }
            }
        }
        //---------------------------------------------------------------------
        private unsafe void AutoCorrelationToArrayImpl(double[] corr, (int Start, int End) range)
        {
#if NET_FULL
            var span = Markers.EnterSpan(1, "Range: {0}", range);
#endif
            this.AddRange(range);

            int n = _values.Length;

            fixed (double* pArray = _values)
            fixed (double* pCorr = corr)
            {
                for (int m = range.Start; m < range.End; ++m)
                {
                    double r_xx = 0;
                    int k       = m;

                    if (Vector.IsHardwareAccelerated && (n - m) >= Vector<double>.Count * 2)
                    {
                        double* a_k  = &pArray[k];
                        double* a_km = pArray;

                        for (; k < n - 2 * Vector<double>.Count; k += 2 * Vector<double>.Count)
                        {
#pragma warning disable CS0618
                            Vector<double> kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            Vector<double> kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);

                            kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);
#pragma warning restore CS0618
                        }
                    }

                    for (; k < n; ++k)
                        r_xx += pArray[k] * pArray[k - m];

                    pCorr[m] = r_xx / (n - m);
                }
            }
#if NET_FULL
            span.Leave();
#endif
        }
    }
}