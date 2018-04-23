using System;
using System.Numerics;
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
            this.CalculateAverageAndVarianceCoreImpl_Base(0, this.N, out double avg, out double var);

            return (avg, var);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Avg, double Var) ISP()
        {
            this.CalculateAverageAndVarianceCoreImpl_ISP(0, this.N, out double avg, out double var);

            return (avg, var);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Avg, double Var) ISP1()
        {
            this.CalculateAverageAndVarianceCoreImpl_ISP1(0, this.N, out double avg, out double var);

            return (avg, var);
        }
        //---------------------------------------------------------------------
        private unsafe void CalculateAverageAndVarianceCoreImpl_Base(int i, int n, out double avg, out double variance)
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
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec            += vec;
                var               += Vector.Dot(vec, vec);
            }
        }
        //---------------------------------------------------------------------
        private unsafe void CalculateAverageAndVarianceCoreImpl_ISP(int i, int n, out double avg, out double variance)
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

                    double var0 = 0;
                    double var1 = 0;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 4 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 5 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 6 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 7 * Vector<double>.Count, ref avgVec1, ref var1, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 1 * Vector<double>.Count, ref avgVec1, ref var1, end);
                        Core(arr, 2 * Vector<double>.Count, ref avgVec0, ref var0, end);
                        Core(arr, 3 * Vector<double>.Count, ref avgVec1, ref var1, end);

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
                    avgVec0 += avgVec1;
                    tmpAvg  += avgVec0.ReduceSum();

                    tmpVariance += var0 + var1;
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
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec            += vec;
                var               += Vector.Dot(vec, vec);
            }
        }
        //---------------------------------------------------------------------
        private unsafe void CalculateAverageAndVarianceCoreImpl_ISP1(int i, int n, out double avg, out double variance)
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
                Vector<double> vec = VectorHelper.GetVector(arr + offset);
                avgVec            += vec;
                var               += vec * vec;
            }
        }
    }
}
