using System;
using System.Diagnostics;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class LoopSimdBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs  = new LoopSimdBenchmarks();
            benchs.N        = 1234;
            const int align = -25;
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Base),align}: {benchs.Base()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Simd_without_advance),align}: {benchs.Simd_without_advance()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.No_unrolling),align}: {benchs.No_unrolling()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Unrolled_4x),align}: {benchs.Unrolled_4x()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Unrolled_4x1),align}: {benchs.Unrolled_4x1()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Unrolled_8x),align}: {benchs.Unrolled_8x()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Unrolled_8x1),align}: {benchs.Unrolled_8x1()}");
            benchs.GlobalSetup();
            Console.WriteLine($"{nameof(benchs.Unrolled_8x2),align}: {benchs.Unrolled_8x2()}");
#if !DEBUG
            BenchmarkRunner.Run<LoopSimdBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(100, 1_000, 10_000, 100_000, 1_000_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private double[]       _values;
        private Vector<double> _res;
        //---------------------------------------------------------------------
        public int Count => _values.Length;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            _values = new double[this.N];
            _res    = new Vector<double>();
            var rnd = new Random(0);

            for (int i = 0; i < this.N; ++i)
                _values[i] = rnd.NextDouble();
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public unsafe double Base()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    Vector<double> res = _res;

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
#pragma warning disable CS0618
                        Vector<double> vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        res += vec;

                        vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        res += vec;
#pragma warning restore CS0618
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double Simd_without_advance()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    Vector<double> res = _res;

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVector(arr);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + Vector<double>.Count);
                        res += vec;

                        arr += 2 * Vector<double>.Count;
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double No_unrolling()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    Vector<double> res = _res;

                    for (; i < n - Vector<double>.Count; i += Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVector(arr);
                        res += vec;

                        arr += Vector<double>.Count;
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double Unrolled_4x()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    Vector<double> res = _res;

                    for (; i < n - 4 * Vector<double>.Count; i += 4 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVector(arr);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 2 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 3 * Vector<double>.Count);
                        res += vec;

                        arr += 4 * Vector<double>.Count;
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double Unrolled_4x1()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    Vector<double> res = _res;

                    for (; i < n - 4 * Vector<double>.Count; i += 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref res);
                        Core(arr, 1 * Vector<double>.Count, ref res);
                        Core(arr, 2 * Vector<double>.Count, ref res);
                        Core(arr, 3 * Vector<double>.Count, ref res);

                        arr += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref res);
                        Core(arr, 1 * Vector<double>.Count, ref res);

                        arr += 2 * Vector<double>.Count;
                        i += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref res);

                        i += Vector<double>.Count;
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double Unrolled_8x()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    Vector<double> res = _res;

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVector(arr);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 2 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 3 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 4 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 5 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 6 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 7 * Vector<double>.Count);
                        res += vec;

                        arr += 8 * Vector<double>.Count;
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double Unrolled_8x1()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    Vector<double> res = _res;

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVector(arr);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 2 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 3 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 4 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 5 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 6 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 7 * Vector<double>.Count);
                        res += vec;

                        arr += 8 * Vector<double>.Count;
                    }

                    for (; i < n - 4 * Vector<double>.Count; i += 4 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVector(arr);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 2 * Vector<double>.Count);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + 3 * Vector<double>.Count);
                        res += vec;

                        arr += 4 * Vector<double>.Count;
                    }

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVector(arr);
                        res += vec;

                        vec = VectorHelper.GetVector(arr + Vector<double>.Count);
                        res += vec;

                        arr += 2 * Vector<double>.Count;
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double Unrolled_8x2()
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    Vector<double> res = _res;

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref res);
                        Core(arr, 1 * Vector<double>.Count, ref res);
                        Core(arr, 2 * Vector<double>.Count, ref res);
                        Core(arr, 3 * Vector<double>.Count, ref res);
                        Core(arr, 4 * Vector<double>.Count, ref res);
                        Core(arr, 5 * Vector<double>.Count, ref res);
                        Core(arr, 6 * Vector<double>.Count, ref res);
                        Core(arr, 7 * Vector<double>.Count, ref res);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref res);
                        Core(arr, 1 * Vector<double>.Count, ref res);
                        Core(arr, 2 * Vector<double>.Count, ref res);
                        Core(arr, 3 * Vector<double>.Count, ref res);

                        arr += 4 * Vector<double>.Count;
                        i += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref res);
                        Core(arr, 1 * Vector<double>.Count, ref res);

                        arr += 2 * Vector<double>.Count;
                        i += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref res);

                        i += Vector<double>.Count;
                    }

                    _res = res;
                }

                for (; i < n; ++i)
                    pArray[i] += 42d;
            }

            return _res[0];
        }
        //---------------------------------------------------------------------
        [DebuggerNonUserCode]
        private static unsafe void Core(double* a, int offset, ref Vector<double> res)
        {
            Vector<double> vec = VectorHelper.GetVector(a + offset);
            res += vec;
        }
    }
}