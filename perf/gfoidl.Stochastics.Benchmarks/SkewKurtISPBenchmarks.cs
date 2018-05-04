using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    public class SkewKurtISPBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs = new SkewKurtISPBenchmarks();
            benchs.N = 1000;
            benchs.GlobalSetup();
            const int align = -35;
            Console.WriteLine($"{nameof(benchs.Base),align}: {benchs.Base()}");
            Console.WriteLine($"{nameof(benchs.ISP),align}: {benchs.ISP()}");
            Console.WriteLine($"{nameof(benchs.ISP1),align}: {benchs.ISP1()}");
#if !DEBUG
            BenchmarkRunner.Run<SkewKurtISPBenchmarks>();
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
        private unsafe void Base(int i, int n, out double skewness, out double kurtosis)
        {
            double tmpSkewness = 0;
            double tmpKurtosis = 0;
            double avg         = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec  = new Vector<double>(avg);
                    var skewVec = Vector<double>.Zero;
                    var kurtVec = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec, ref kurtVec, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    tmpSkewness += skewVec.ReduceSum();
                    tmpKurtosis += kurtVec.ReduceSum();
                }

                while (arr < end)
                {
                    double t     = *arr - avg;
                    double t1    = t * t * t;
                    tmpSkewness += t1;
                    tmpKurtosis += t1 * t;
                    arr++;
                }

                skewness = tmpSkewness;
                kurtosis = tmpKurtosis;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec, ref Vector<double> kurtVec, double* end)
            {
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                vec               -= avgVec;
                Vector<double> tmp = vec * vec * vec;
                skewVec           += tmp;
                kurtVec           += tmp * vec;
            }
        }
        //---------------------------------------------------------------------
        private unsafe void ISP(int i, int n, out double skewness, out double kurtosis)
        {
            double tmpSkewness = 0;
            double tmpKurtosis = 0;
            double avg         = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec   = new Vector<double>(avg);
                    var skewVec0 = Vector<double>.Zero;
                    var skewVec1 = Vector<double>.Zero;
                    var skewVec2 = Vector<double>.Zero;
                    var skewVec3 = Vector<double>.Zero;

                    var kurtVec0 = Vector<double>.Zero;
                    var kurtVec1 = Vector<double>.Zero;
                    var kurtVec2 = Vector<double>.Zero;
                    var kurtVec3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref kurtVec2, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec3, ref kurtVec3, end);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref skewVec2, ref kurtVec2, end);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref skewVec3, ref kurtVec3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref kurtVec2, end);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref skewVec3, ref kurtVec3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref skewVec1, ref kurtVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    skewVec0    += skewVec1 + skewVec2 + skewVec3;
                    tmpSkewness += skewVec0.ReduceSum();

                    kurtVec0    += kurtVec1 + kurtVec2 + kurtVec3;
                    tmpKurtosis += kurtVec0.ReduceSum();
                }

                while (arr < end)
                {
                    double t     = *arr - avg;
                    double t1    = t * t * t;
                    tmpSkewness += t1;
                    tmpKurtosis += t1 * t;
                    arr++;
                }

                skewness = tmpSkewness;
                kurtosis = tmpKurtosis;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec, ref Vector<double> kurtVec, double* end)
            {
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                vec               -= avgVec;
                Vector<double> tmp = vec * vec * vec;
                skewVec           += tmp;
                kurtVec           += tmp * vec;
            }
        }
        //---------------------------------------------------------------------
        private unsafe void ISP1(int i, int n, out double skewness, out double kurtosis)
        {
            double tmpSkewness = 0;
            double tmpKurtosis = 0;
            double avg         = this.Mean;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var avgVec   = new Vector<double>(avg);
                    var skewVec0 = Vector<double>.Zero;
                    var skewVec1 = Vector<double>.Zero;
                    var skewVec2 = Vector<double>.Zero;
                    var skewVec3 = Vector<double>.Zero;

                    var kurtVec0 = Vector<double>.Zero;
                    var kurtVec1 = Vector<double>.Zero;
                    var kurtVec2 = Vector<double>.Zero;
                    var kurtVec3 = Vector<double>.Zero;

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        ISP1Core2(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);
                        ISP1Core2(arr, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref skewVec3, ref kurtVec2, ref kurtVec3, end);
                        ISP1Core2(arr, 4 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);
                        ISP1Core2(arr, 6 * Vector<double>.Count, avgVec, ref skewVec2, ref skewVec3, ref kurtVec2, ref kurtVec3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core2(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);
                        ISP1Core2(arr, 2 * Vector<double>.Count, avgVec, ref skewVec2, ref skewVec3, ref kurtVec2, ref kurtVec3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        ISP1Core2(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref skewVec1, ref kurtVec0, ref kurtVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref skewVec0, ref kurtVec0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction -- https://github.com/gfoidl/Stochastics/issues/43
                    skewVec0    += skewVec1;
                    skewVec2    += skewVec3;
                    skewVec0    += skewVec2;
                    tmpSkewness += skewVec0.ReduceSum();

                    kurtVec0    += kurtVec1;
                    kurtVec2    += kurtVec3;
                    kurtVec0    += kurtVec2;
                    tmpKurtosis += kurtVec0.ReduceSum();
                }

                while (arr < end)
                {
                    double t     = *arr - avg;
                    double t1    = t * t * t;
                    tmpSkewness += t1;
                    tmpKurtosis += t1 * t;
                    arr++;
                }

                skewness = tmpSkewness;
                kurtosis = tmpKurtosis;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec, ref Vector<double> kurtVec, double* end)
            {
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                vec               -= avgVec;
                Vector<double> tmp = vec * vec * vec;
                skewVec           += tmp;
                kurtVec           += tmp * vec;
            }
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ISP1Core2(double* arr, int offset, Vector<double> avgVec, ref Vector<double> skewVec0, ref Vector<double> skewVec1, ref Vector<double> kurtVec0, ref Vector<double> kurtVec1, double* end)
        {
            Vector<double> vec0 = VectorHelper.GetVectorUnaligned(arr + offset + 0 * Vector<double>.Count);
            Vector<double> vec1 = VectorHelper.GetVectorUnaligned(arr + offset + 1 * Vector<double>.Count);

            vec0 -= avgVec;
            vec1 -= avgVec;

            var tmp0 = vec0 * vec0;
            var tmp1 = vec1 * vec1;

            tmp0 *= vec0;
            tmp1 *= vec1;

            skewVec0 += tmp0;
            skewVec1 += tmp1;

            tmp0 *= vec0;
            tmp1 *= vec1;

            kurtVec0 += tmp0;
            kurtVec1 += tmp1;
        }
    }
}
