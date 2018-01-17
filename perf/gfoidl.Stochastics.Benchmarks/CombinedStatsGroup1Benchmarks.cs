//#define SEQUENTIAL
//-----------------------------------------------------------------------------
using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class CombinedStatsGroup1Benchmarks
    {
        public static void Run()
        {
            var benchs      = new CombinedStatsGroup1Benchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.EachSeparate),align}: {benchs.EachSeparate()}");
            Console.WriteLine($"{nameof(benchs.SimdSequential),align}: {benchs.SimdSequential()}");
            Console.WriteLine($"{nameof(benchs.SimdParallel),align}: {benchs.SimdParallel()}");
#if !DEBUG
            BenchmarkRunner.Run<CombinedStatsGroup1Benchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(100, 1_000, 10_000, 50_000, 100_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private double[] _values;
        //---------------------------------------------------------------------
        public int Count => _values.Length;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            _values = new double[this.N];
            var rnd = new Random(0);

            for (int i = 0; i < this.N; ++i)
                _values[i] = rnd.NextDouble();
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public (double Min, double Max) EachSeparate()
        {
            var sample = new Sample(_values);

            double min = sample.Min;
            double max = sample.Max;

            return (min, max);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Min, double Max) SimdSequential()
        {
            var (min, max) = this.CombinedImpl((0, this.Count));

            return (min, max);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Min, double Max) SimdParallel()
        {
            double min = double.MaxValue;
            double max = double.MinValue;

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);
#if SEQUENTIAL
            (double Min, double Max) tmp = this.CombinedImpl((0, this.Count));
            tmp.Min.InterlockedExchangeIfSmaller(ref min, tmp.Min);
            tmp.Max.InterlockedExchangeIfGreater(ref max, tmp.Max);
#else
            Parallel.ForEach(
                partitioner,
                range =>
                {
                    (double Min, double Max) tmp = this.CombinedImpl((range.Item1, range.Item2));
                    tmp.Min.InterlockedExchangeIfSmaller(ref min, tmp.Min);
                    tmp.Max.InterlockedExchangeIfGreater(ref max, tmp.Max);
                }
            );
#endif
            return (min, max);
        }
        //---------------------------------------------------------------------
        private unsafe (double Min, double Max) CombinedImpl((int start, int end) range)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            var (i, n) = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var minVec = new Vector<double>(double.MaxValue);
                    var maxVec = new Vector<double>(double.MinValue);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 2 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 3 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 4 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 5 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 6 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 7 * Vector<double>.Count, ref minVec, ref maxVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 2 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 3 * Vector<double>.Count, ref minVec, ref maxVec);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                    {
                        min = Math.Min(min, minVec[j]);
                        max = Math.Max(max, maxVec[j]);
                    }
                }

                for (; i < n; ++i)
                {
                    min = Math.Min(min, pArray[i]);
                    max = Math.Max(max, pArray[i]);
                }
            }

            return (min, max);
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> minVec, ref Vector<double> maxVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                minVec             = Vector.Min(minVec, vec);
                maxVec             = Vector.Max(maxVec, vec);
            }
        }
    }
}