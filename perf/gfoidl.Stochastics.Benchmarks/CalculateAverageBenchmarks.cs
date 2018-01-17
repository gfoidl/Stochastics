using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class CalculateAverageBenchmarks
    {
        public static void Run()
        {
            var benchs      = new CalculateAverageBenchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.Linq),align}: {benchs.Linq()}");
            Console.WriteLine($"{nameof(benchs.PLinq),align}: {benchs.PLinq()}");
            Console.WriteLine($"{nameof(benchs.UnsafeSimd),align}: {benchs.UnsafeSimd()}");
            Console.WriteLine($"{nameof(benchs.ParallelizedUnsafeSimd),align}: {benchs.ParallelizedUnsafeSimd()}");
#if !DEBUG
            BenchmarkRunner.Run<CalculateAverageBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(100, 1_000, 10_000, 100_000, 1_000_000)]
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
        //[Benchmark(Baseline = true)]
        public double Linq() => _values.Average();
        //---------------------------------------------------------------------
        //[Benchmark]
        public double PLinq() => _values.AsParallel().Average();
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double UnsafeSimd()
        {
            double avg = this.CalculateAverageImpl((0, this.Count));

            return avg / this.Count;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double ParallelizedUnsafeSimd()
        {
            double avg = 0;

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);

            Parallel.ForEach(
                partitioner,
                range =>
                {
                    double tmp = this.CalculateAverageImpl((range.Item1, range.Item2));
                    tmp.SafeAdd(ref avg);
                }
            );

            return avg / this.Count;
        }
        //---------------------------------------------------------------------
        private unsafe double CalculateAverageImpl((int start, int end) range)
        {
            double avg = 0;
            var (i, n) = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec = new Vector<double>(avg);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        avg += avgVec[j];
                }

                for (; i < n; ++i)
                    avg += pArray[i];
            }

            return avg;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec += vec;
            }
        }
    }
}