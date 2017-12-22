using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    //[DisassemblyDiagnoser(printSource: true)]
    public class CalculateDeltaBenchmarks
    {
        public static void Run()
        {
            var benchs = new CalculateDeltaBenchmarks();
            benchs.N   = 1000;
            benchs.GlobalSetup();
            Console.WriteLine(benchs.Sequential());
            Console.WriteLine(benchs.Simd());
            Console.WriteLine(benchs.Parallelized());
            Console.WriteLine(benchs.ParallelizedSimd());
#if !DEBUG
            BenchmarkRunner.Run<CalculateDeltaBenchmarks>();
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
            double delta = 0;
            double[] tmp = _values;
            double avg   = this.Mean;

            for (int i = 0; i < tmp.Length; ++i)
                delta += Math.Abs(tmp[i] - avg);

            return delta / tmp.Length;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double Simd()
        {
            double delta = 0;
            double[] tmp = _values;
            double avg   = this.Mean;
            int i        = 0;

            if (Vector.IsHardwareAccelerated && this.Count >= Vector<double>.Count * 2)
            {
                var avgVec   = new Vector<double>(avg);
                var deltaVec = new Vector<double>(0);

                for (; i < tmp.Length - 2 * Vector<double>.Count; i += Vector<double>.Count)
                {
                    var tmpVec = new Vector<double>(tmp, i);
                    deltaVec += Vector.Abs(tmpVec - avgVec);

                    i += Vector<double>.Count;
                    tmpVec = new Vector<double>(tmp, i);
                    deltaVec += Vector.Abs(tmpVec - avgVec);
                }

                for (int j = 0; j < Vector<double>.Count; ++j)
                    delta += deltaVec[j];
            }

            for (; i < tmp.Length; ++i)
                delta += Math.Abs(tmp[i] - avg);

            return delta / tmp.Length;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double Parallelized()
        {
            double delta = 0;
            var sync     = new object();

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);

            Parallel.ForEach(
                partitioner,
                range =>
                {
                    double localDelta = 0;
                    double[] arr      = _values;
                    double avg        = this.Mean;

                    // RCE
                    if ((uint)range.Item1 >= arr.Length || (uint)range.Item2 > arr.Length)
                        ThrowHelper.ThrowArgumentOutOfRange(nameof(range));

                    for (int i = range.Item1; i < range.Item2; ++i)
                        localDelta += Math.Abs(arr[i] - avg);

                    lock (sync) delta += localDelta;
                }
            );

            return delta / _values.Length;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double ParallelizedSimd()
        {
            double delta = 0;
            var sync     = new object();

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);
#if SEQUENTIAL
            var range = Tuple.Create(0, _values.Length);
#else
            Parallel.ForEach(
                partitioner,
                range =>
                {
#endif
                    double localDelta = 0;
                    double[] arr      = _values;
                    double avg        = this.Mean;
                    int i             = range.Item1;

                    // RCE
                    if ((uint)range.Item1 >= arr.Length || (uint)range.Item2 > arr.Length)
                        ThrowHelper.ThrowArgumentOutOfRange(nameof(range));

                    if (Vector.IsHardwareAccelerated && (range.Item2 - range.Item1) >= Vector<double>.Count * 2)
                    {
                        var avgVec   = new Vector<double>(avg);
                        var deltaVec = new Vector<double>(0);

                        for (; i < range.Item2 - 2 * Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var arrVec = new Vector<double>(arr, i);
                            deltaVec += Vector.Abs(arrVec - avgVec);

                            i += Vector<double>.Count;
                            arrVec = new Vector<double>(arr, i);
                            deltaVec += Vector.Abs(arrVec - avgVec);
                        }

                        for (int j = 0; j < Vector<double>.Count; ++j)
                            localDelta += deltaVec[j];
                    }

                    for (; i < range.Item2; ++i)
                        localDelta += Math.Abs(arr[i] - avg);

                    lock (sync) delta += localDelta;
#if !SEQUENTIAL
                }
            );
#endif
            return delta / _values.Length;
        }
    }
}