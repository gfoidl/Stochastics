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
    public class CombinedStatsGroup2Benchmarks
    {
        public static void Run()
        {
            var benchs      = new CombinedStatsGroup2Benchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.EachSeparate),align}: {benchs.EachSeparate()}");
            Console.WriteLine($"{nameof(benchs.Combined),align}: {benchs.Combined()}");
            Console.WriteLine($"{nameof(benchs.CombinedSequential),align}: {benchs.CombinedSequential()}");
            Console.WriteLine($"{nameof(benchs.CombinedParallel),align}: {benchs.CombinedParallel()}");
#if !DEBUG
            BenchmarkRunner.Run<CombinedStatsGroup2Benchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(100, 1_000, 10_000, 100_000)]
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
        public (double Mean, double Variance) EachSeparate()
        {
            var sample = new Sample(_values);

            double avg      = sample.Mean;
            double variance = sample.Variance;

            return (avg, variance);
        }
        //---------------------------------------------------------------------
        //[Benchmark]
        public (double Mean, double Variance) Combined()
        {
            return this.Count < Sample.ThresholdForParallel
                ? this.CombinedSequential()
                : this.CombinedParallel();
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Mean, double Variance) CombinedSequential()
        {
            var (avg, variance) = this.CombinedImpl((0, this.Count));

            avg      /= this.Count;
            variance -= this.Count * avg * avg;
            variance /= this.Count;

            return (avg, variance);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Mean, double Variance) CombinedParallel()
        {
            double avg      = 0;
            double variance = 0;

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);
#if SEQUENTIAL
            (double Mean, double Variance) tmp = this.CombinedImpl((0, this.Count));
            avg      = tmp.Mean;
            variance = tmp.Variance;
#else
            Parallel.ForEach(
                partitioner,
                range =>
                {
                    (double Mean, double Variance) tmp = this.CombinedImpl((range.Item1, range.Item2));
                    tmp.Mean.SafeAdd(ref avg);
                    tmp.Variance.SafeAdd(ref variance);
                }
            );
#endif
            avg      /= this.Count;
            variance -= this.Count * avg * avg;
            variance /= this.Count;

            return (avg, variance);
        }
        //---------------------------------------------------------------------
        private unsafe (double Mean, double Variance) CombinedImpl((int start, int end) range)
        {
            double avg      = 0;
            double variance = 0;
            var (i, n)      = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec = new Vector<double>(avg);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec, ref variance);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref variance);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref variance);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref variance);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        avg += avgVec[j];
                }

                for (; i < n; ++i)
                {
                    avg      += pArray[i];
                    variance += pArray[i] * pArray[i];
                }
            }

            return (avg, variance);
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec, ref double var)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec += vec;
                var    += Vector.Dot(vec, vec);
            }
        }
    }
}