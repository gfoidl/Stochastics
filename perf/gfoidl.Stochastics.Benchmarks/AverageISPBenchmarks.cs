using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    public class AverageISPBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs = new AverageISPBenchmarks();
            benchs.N = 1000;
            benchs.GlobalSetup();
            const int align = -35;
            Console.WriteLine($"{nameof(benchs.Base),align}: {benchs.Base()}");
            Console.WriteLine($"{nameof(benchs.ISP),align}: {benchs.ISP()}");
            Console.WriteLine($"{nameof(benchs.ISP1),align}: {benchs.ISP1()}");
#if !DEBUG
            BenchmarkRunner.Run<AverageISPBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(100, 1_000, 10_000, 100_000, 1_000_000)]
        public int N { get; set; } = 10_123;
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
        public (double Avg, double Var) Base()
        {
            this.Base(0, this.N, out double avg, out double var);

            return (avg, var);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Avg, double Var) ISP()
        {
            this.ISP(0, this.N, out double avg, out double var);

            return (avg, var);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Avg, double Var) ISP1()
        {
            this.ISP1(0, this.N, out double avg, out double var);

            return (avg, var);
        }
        //---------------------------------------------------------------------
        private unsafe void Base(int i, int n, out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec, ref tmpVariance, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    tmpAvg += avgVec.ReduceSum();
                }

                while (arr < end)
                {
                    tmpAvg      += *arr;
                    tmpVariance += *arr * *arr;
                    arr++;
                }

                avg      = tmpAvg;
                variance = tmpVariance;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec, ref double var, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                avgVec            += vec;
                var               += Vector.Dot(vec, vec);
            }
        }
        //---------------------------------------------------------------------
        private unsafe void ISP(int i, int n, out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec0 = Vector<double>.Zero;
                    var avgVec1 = Vector<double>.Zero;
                    var avgVec2 = Vector<double>.Zero;
                    var avgVec3 = Vector<double>.Zero;

                    var var0 = Vector<double>.Zero;
                    var var1 = Vector<double>.Zero;
                    var var2 = Vector<double>.Zero;
                    var var3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec3, ref var3, end);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec3, ref var3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec2, ref var2, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec3, ref var3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    avgVec0 += avgVec1 + avgVec2 + avgVec3;
                    tmpAvg  += avgVec0.ReduceSum();

                    var0        += var1 + var2 + var3;
                    tmpVariance += var0.ReduceSum();
                }

                while (arr < end)
                {
                    tmpAvg      += *arr;
                    tmpVariance += *arr * *arr;
                    arr++;
                }

                avg      = tmpAvg;
                variance = tmpVariance;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> avgVec, ref Vector<double> var, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                avgVec            += vec;
                var               += vec * vec;
            }
        }
        //---------------------------------------------------------------------
        private unsafe void ISP1(int i, int n, out double avg, out double variance)
        {
            double tmpAvg      = 0;
            double tmpVariance = 0;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec0 = Vector<double>.Zero;
                    var avgVec1 = Vector<double>.Zero;
                    var avgVec2 = Vector<double>.Zero;
                    var avgVec3 = Vector<double>.Zero;

                    var var0 = Vector<double>.Zero;
                    var var1 = Vector<double>.Zero;
                    var var2 = Vector<double>.Zero;
                    var var3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        ISP1Core4(arr, 0 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref avgVec2, ref avgVec3, ref var0, ref var1, ref var2, ref var3, end);
                        ISP1Core4(arr, 4 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref avgVec2, ref avgVec3, ref var0, ref var1, ref var2, ref var3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core4(arr, 0 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref avgVec2, ref avgVec3, ref var0, ref var1, ref var2, ref var3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core2(arr, 0 * Vector<double>.Count, ref avgVec0, ref avgVec1, ref var0, ref var1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    avgVec0 += avgVec1;
                    avgVec2 += avgVec3;
                    avgVec0 += avgVec2;
                    tmpAvg  += avgVec0.ReduceSum();

                    var0        += var1;
                    var2        += var3;
                    var0        += var2;
                    tmpVariance += var0.ReduceSum();
                }

                while (arr < end)
                {
                    tmpAvg      += *arr;
                    tmpVariance += *arr * *arr;
                    arr++;
                }

                avg      = tmpAvg;
                variance = tmpVariance;
            }
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ISP1Core4(double* arr, int offset, ref Vector<double> avgVec0, ref Vector<double> avgVec1, ref Vector<double> avgVec2, ref Vector<double> avgVec3, ref Vector<double> var0, ref Vector<double> var1, ref Vector<double> var2, ref Vector<double> var3, double* end)
        {
            Vector<double> vec0 = VectorHelper.GetVectorUnaligned(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVectorUnaligned(arr + offset + 1 * Vector<double>.Count);
            Vector<double> vec2 = VectorHelper.GetVectorUnaligned(arr + offset + 2 * Vector<double>.Count);
            Vector<double> vec3 = VectorHelper.GetVectorUnaligned(arr + offset + 3 * Vector<double>.Count);

            avgVec0 += vec0;
            avgVec1 += vec1;
            avgVec2 += vec2;
            avgVec3 += vec3;

            var tmp0 = vec0 * vec0;
            var tmp1 = vec1 * vec1;
            var tmp2 = vec2 * vec2;
            var tmp3 = vec3 * vec3;

            var0 += tmp0;
            var1 += tmp1;
            var2 += tmp2;
            var3 += tmp3;
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ISP1Core2(double* arr, int offset, ref Vector<double> avgVec0, ref Vector<double> avgVec1, ref Vector<double> var0, ref Vector<double> var1, double* end)
        {
            Vector<double> vec0 = VectorHelper.GetVectorUnaligned(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVectorUnaligned(arr + offset + 1 * Vector<double>.Count);

            avgVec0 += vec0;
            avgVec1 += vec1;

            var tmp0 = vec0 * vec0;
            var tmp1 = vec1 * vec1;

            var0 += tmp0;
            var1 += tmp1;
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ISP1Core(double* arr, int offset, ref Vector<double> avgVec, ref Vector<double> var, double* end)
        {
            Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
            avgVec            += vec;
            var               += vec * vec;
        }
    }
}
