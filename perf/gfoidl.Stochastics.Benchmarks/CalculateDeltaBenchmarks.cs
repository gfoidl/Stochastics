using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class CalculateDeltaBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs = new CalculateDeltaBenchmarks();
            benchs.N   = 1000;
            benchs.GlobalSetup();
            const int align = -35;
            Console.WriteLine($"{nameof(benchs.Sequential),align}: {benchs.Sequential()}");
            Console.WriteLine($"{nameof(benchs.Simd),align}: {benchs.Simd()}");
            Console.WriteLine($"{nameof(benchs.UnsafeSimd),align}: {benchs.UnsafeSimd()}");
            Console.WriteLine($"{nameof(benchs.Parallelized),align}: {benchs.Parallelized()}");
            Console.WriteLine($"{nameof(benchs.ParallelizedSimd),align}: {benchs.ParallelizedSimd()}");
            Console.WriteLine($"{nameof(benchs.ParallelizedUnsafeSimd),align}: {benchs.ParallelizedUnsafeSimd()}");
            Console.WriteLine($"{nameof(benchs.ParallelizedUnsafeSimdUnrolled),align}: {benchs.ParallelizedUnsafeSimdUnrolled()}");
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
                _avg      += _values[i];
            }

            _avg /= this.N;
        }
        //---------------------------------------------------------------------
        //[Benchmark(Baseline = true)]
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
        //[Benchmark]
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
                    deltaVec  += Vector.Abs(tmpVec - avgVec);

                    i        += Vector<double>.Count;
                    tmpVec    = new Vector<double>(tmp, i);
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
        //[Benchmark]
        public unsafe double UnsafeSimd()
        {
            double delta = 0;
            double avg   = this.Mean;
            int n        = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    var avgVec   = new Vector<double>(avg);
                    var deltaVec = new Vector<double>(0);

                    for (; i < n - 2 * Vector<double>.Count;)
                    {
                        Vector<double> vec = Unsafe.Read<Vector<double>>(arr);
                        deltaVec += Vector.Abs(vec - avgVec);
                        arr      += Vector<double>.Count;
                        i        += Vector<double>.Count;

                        vec = Unsafe.Read<Vector<double>>(arr);
                        deltaVec += Vector.Abs(vec - avgVec);
                        arr      += Vector<double>.Count;
                        i        += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        delta += deltaVec[j];
                }

                for (; i < n; ++i)
                    delta += Math.Abs(pArray[i] - avg);
            }

            return delta / n;
        }
        //---------------------------------------------------------------------
        //[Benchmark]
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
                        ThrowHelper.ThrowArgumentOutOfRange(ThrowHelper.ExceptionArgument.range);

                    for (int i = range.Item1; i < range.Item2; ++i)
                        localDelta += Math.Abs(arr[i] - avg);

                    lock (sync) delta += localDelta;
                }
            );

            return delta / _values.Length;
        }
        //---------------------------------------------------------------------
        //[Benchmark]
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
                        ThrowHelper.ThrowArgumentOutOfRange(ThrowHelper.ExceptionArgument.range);

                    if (Vector.IsHardwareAccelerated && (range.Item2 - range.Item1) >= Vector<double>.Count * 2)
                    {
                        var avgVec   = new Vector<double>(avg);
                        var deltaVec = new Vector<double>(0);

                        for (; i < range.Item2 - 2 * Vector<double>.Count; i += Vector<double>.Count)
                        {
                            var arrVec = new Vector<double>(arr, i);
                            deltaVec  += Vector.Abs(arrVec - avgVec);

                            i        += Vector<double>.Count;
                            arrVec    = new Vector<double>(arr, i);
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
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public unsafe double ParallelizedUnsafeSimd()
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
                    double avg        = this.Mean;
                    int n             = range.Item2;

                    fixed (double* pArray = _values)
                    {
                        int i         = range.Item1;
                        //double* arr = &pArray[i];
                        double* arr   = pArray + i;   // is the same as above

                        if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count * 2)
                        {
                            var avgVec   = new Vector<double>(avg);
                            var deltaVec = new Vector<double>(0);

                            for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                            {
                                Vector<double> vec = Unsafe.Read<Vector<double>>(arr);
                                deltaVec += Vector.Abs(vec - avgVec);
                                arr      += Vector<double>.Count;

                                vec = Unsafe.Read<Vector<double>>(arr);
                                deltaVec += Vector.Abs(vec - avgVec);
                                arr      += Vector<double>.Count;
                            }

                            for (int j = 0; j < Vector<double>.Count; ++j)
                                localDelta += deltaVec[j];
                        }

                        for (; i < n; ++i)
                            localDelta += Math.Abs(pArray[i] - avg);
                    }

                    lock (sync) delta += localDelta;
#if !SEQUENTIAL
                }
            );
#endif
            return delta / _values.Length;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double ParallelizedUnsafeSimdUnrolled()
        {
            double delta = 0;
            var sync = new object();

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
                    double avg        = this.Mean;
                    int n             = range.Item2;

                    fixed (double* pArray = _values)
                    {
                        int i = range.Item1;
                        //double* arr = &pArray[i];
                        double* arr = pArray + i;     // is the same as above

                        if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                        {
                            var avgVec   = new Vector<double>(avg);
                            var deltaVec = new Vector<double>(0);

                            for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                            {
                                Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 4 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 5 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 6 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 7 * Vector<double>.Count, avgVec, ref deltaVec);

                                arr += 8 * Vector<double>.Count;
                            }

                            if (i < n - 4 * Vector<double>.Count)
                            {
                                Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec);

                                arr += 4 * Vector<double>.Count;
                                i   += 4 * Vector<double>.Count;
                            }

                            if (i < n - 2 * Vector<double>.Count)
                            {
                                Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);
                                Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec);

                                arr += 2 * Vector<double>.Count;
                                i   += 2 * Vector<double>.Count;
                            }

                            if (i < n - Vector<double>.Count)
                            {
                                Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec);

                                i += Vector<double>.Count;
                            }

                            for (int j = 0; j < Vector<double>.Count; ++j)
                                localDelta += deltaVec[j];
                        }

                        for (; i < n; ++i)
                            localDelta += Math.Abs(pArray[i] - avg);
                    }

                    lock (sync) delta += localDelta;
#if !SEQUENTIAL
                }
            );
#endif
            return delta / _values.Length;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec)
            {
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                deltaVec += Vector.Abs(vec - avgVec);
            }
        }
    }
}
