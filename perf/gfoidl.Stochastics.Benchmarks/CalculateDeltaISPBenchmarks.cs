using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    public class CalculateDeltaISPBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs = new CalculateDeltaISPBenchmarks();
            benchs.N = 123;
            benchs.GlobalSetup();
            const int align = -35;
            Console.WriteLine($"{nameof(benchs.Base),align}: {benchs.Base()}");
            Console.WriteLine($"{nameof(benchs.ISP),align}: {benchs.ISP()}");
            Console.WriteLine($"{nameof(benchs.ISP1),align}: {benchs.ISP1()}");
#if !DEBUG
            BenchmarkRunner.Run<CalculateDeltaISPBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(100, 1_000, 10_000, 100_000, 1_000_000)]
        public int N { get; set; } = 10_123;
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
        [Benchmark(Baseline = true)]
        public unsafe double Base()
        {
            int i = 0;
            int n = this.N;

            double delta = 0;
            double avg   = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec   = new Vector<double>(avg);
                    var deltaVec = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref deltaVec, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    delta += deltaVec.ReduceSum();
                }

                while (arr < end)
                {
                    delta += Math.Abs(*arr - avg);
                    arr++;
                }
            }

            return delta;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                deltaVec          += Vector.Abs(vec - avgVec);
            }
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double ISP()
        {
            int i = 0;
            int n = this.N;

            double delta = 0;
            double avg   = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec    = new Vector<double>(avg);
                    var deltaVec0 = Vector<double>.Zero;
                    var deltaVec1 = Vector<double>.Zero;
                    var deltaVec2 = Vector<double>.Zero;
                    var deltaVec3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec3, end);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref deltaVec3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref deltaVec2, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref deltaVec3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref deltaVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    deltaVec0 += deltaVec1 + deltaVec2 + deltaVec3;
                    delta     += deltaVec0.ReduceSum();
                }

                while (arr < end)
                {
                    delta += Math.Abs(*arr - avg);
                    arr++;
                }
            }

            return delta;
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                deltaVec += Vector.Abs(vec - avgVec);
            }
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double ISP1()
        {
            int i = 0;
            int n = this.N;

            double delta = 0;
            double avg   = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec    = new Vector<double>(avg);
                    var deltaVec0 = Vector<double>.Zero;
                    var deltaVec1 = Vector<double>.Zero;
                    var deltaVec2 = Vector<double>.Zero;
                    var deltaVec3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        ISP1Core4(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, ref deltaVec2, ref deltaVec3, end);
                        ISP1Core4(arr, 4 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, ref deltaVec2, ref deltaVec3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core4(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, ref deltaVec2, ref deltaVec3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core2(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, ref deltaVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core(arr, 0 * Vector<double>.Count, avgVec, ref deltaVec0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    deltaVec0 += deltaVec1;
                    deltaVec2 += deltaVec3;
                    deltaVec0 += deltaVec2;
                    delta     += deltaVec0.ReduceSum();
                }

                while (arr < end)
                {
                    delta += Math.Abs(*arr - avg);
                    arr++;
                }
            }

            return delta;
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ISP1Core4(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec0, ref Vector<double> deltaVec1, ref Vector<double> deltaVec2, ref Vector<double> deltaVec3, double* end)
        {
            Vector<double> vec0 = VectorHelper.GetVectorUnaligned(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVectorUnaligned(arr + offset + 1 * Vector<double>.Count);
            Vector<double> vec2 = VectorHelper.GetVectorUnaligned(arr + offset + 2 * Vector<double>.Count);
            Vector<double> vec3 = VectorHelper.GetVectorUnaligned(arr + offset + 3 * Vector<double>.Count);

            vec0 -= avgVec;
            vec1 -= avgVec;
            vec2 -= avgVec;
            vec3 -= avgVec;

            deltaVec0 += Vector.Abs(vec0);
            deltaVec1 += Vector.Abs(vec1);
            deltaVec2 += Vector.Abs(vec2);
            deltaVec3 += Vector.Abs(vec3);
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ISP1Core2(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec0, ref Vector<double> deltaVec1, double* end)
        {
            Vector<double> vec0 = VectorHelper.GetVectorUnaligned(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVectorUnaligned(arr + offset + 1 * Vector<double>.Count);

            vec0 -= avgVec;
            vec1 -= avgVec;

            deltaVec0 += Vector.Abs(vec0);
            deltaVec1 += Vector.Abs(vec1);
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ISP1Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> deltaVec, double* end)
        {
            Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
            deltaVec          += Vector.Abs(vec - avgVec);
        }
    }
}
