using System;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    public class MinMaxISPBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs = new MinMaxISPBenchmarks();
            benchs.N = 1000;
            benchs.GlobalSetup();
            const int align = -35;
            Console.WriteLine($"{nameof(benchs.Base),align}: {benchs.Base()}");
            Console.WriteLine($"{nameof(benchs.ISP),align}: {benchs.ISP()}");
            Console.WriteLine($"{nameof(benchs.ISP1),align}: {benchs.ISP1()}");
#if !DEBUG
            BenchmarkRunner.Run<MinMaxISPBenchmarks>();
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
        public (double Min, double Max) Base()
        {
            this.GetMinMaxImpl_Base(0, this.N, out double min, out double max);

            return (min, max);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Min, double Max) ISP()
        {
            this.GetMinMaxImpl_ISP(0, this.N, out double min, out double max);

            return (min, max);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Min, double Max) ISP1()
        {
            this.GetMinMaxImpl_ISP1(0, this.N, out double min, out double max);

            return (min, max);
        }
        //---------------------------------------------------------------------
        private unsafe void GetMinMaxImpl_Base(int i, int n, out double min, out double max)
        {
            double tmpMin = double.MaxValue;
            double tmpMax = double.MinValue;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var minVec = new Vector<double>(tmpMin);
                    var maxVec = new Vector<double>(tmpMax);

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 4 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 5 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 6 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 7 * Vector<double>.Count, ref minVec, ref maxVec, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec, ref maxVec, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec, ref maxVec, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec, ref maxVec, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction
                    VectorHelper.ReduceMinMax(minVec, maxVec, ref tmpMin, ref tmpMax);
                }

                while (arr < end)
                {
                    if (*arr < tmpMin) tmpMin = *arr;
                    if (*arr > tmpMax) tmpMax = *arr;
                    arr++;
                }

                min = tmpMin;
                max = tmpMax;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> minVec, ref Vector<double> maxVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                minVec = Vector.Min(minVec, vec);
                maxVec = Vector.Max(maxVec, vec);
            }
        }
        //---------------------------------------------------------------------
        private unsafe void GetMinMaxImpl_ISP(int i, int n, out double min, out double max)
        {
            double tmpMin = double.MaxValue;
            double tmpMax = double.MinValue;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var minVec0 = new Vector<double>(tmpMin);
                    var minVec1 = new Vector<double>(tmpMin);
                    var maxVec0 = new Vector<double>(tmpMax);
                    var maxVec1 = new Vector<double>(tmpMax);

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 4 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 5 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 6 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 7 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction
                    minVec0 = Vector.Min(minVec0, minVec1);
                    maxVec0 = Vector.Max(maxVec0, maxVec1);
                    VectorHelper.ReduceMinMax(minVec0, maxVec0, ref tmpMin, ref tmpMax);
                }

                while (arr < end)
                {
                    if (*arr < tmpMin) tmpMin = *arr;
                    if (*arr > tmpMax) tmpMax = *arr;
                    arr++;
                }

                min = tmpMin;
                max = tmpMax;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> minVec, ref Vector<double> maxVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                minVec = Vector.Min(minVec, vec);
                maxVec = Vector.Max(maxVec, vec);
            }
        }
        //---------------------------------------------------------------------
        private unsafe void GetMinMaxImpl_ISP1(int i, int n, out double min, out double max)
        {
            double tmpMin = double.MaxValue;
            double tmpMax = double.MinValue;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;
                n          -= i;
                i           = 0;
                double* end = arr + n;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var minVec0 = new Vector<double>(tmpMin);
                    var minVec1 = new Vector<double>(tmpMin);
                    var minVec2 = new Vector<double>(tmpMin);
                    var minVec3 = new Vector<double>(tmpMin);

                    var maxVec0 = new Vector<double>(tmpMax);
                    var maxVec1 = new Vector<double>(tmpMax);
                    var maxVec2 = new Vector<double>(tmpMax);
                    var maxVec3 = new Vector<double>(tmpMax);

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec3, ref maxVec3, end);
                        Core(arr, 4 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 5 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 6 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(arr, 7 * Vector<double>.Count, ref minVec3, ref maxVec3, end);

                        arr += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);
                        Core(arr, 2 * Vector<double>.Count, ref minVec2, ref maxVec2, end);
                        Core(arr, 3 * Vector<double>.Count, ref minVec3, ref maxVec3, end);

                        arr += 4 * Vector<double>.Count;
                        i   += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec0, ref maxVec0, end);
                        Core(arr, 1 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        arr += 2 * Vector<double>.Count;
                        i   += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(arr, 0 * Vector<double>.Count, ref minVec1, ref maxVec1, end);

                        arr += 1 * Vector<double>.Count;
                    }

                    // Reduction
                    minVec0 = Vector.Min(minVec0, Vector.Min(minVec1, Vector.Min(minVec2, minVec3)));
                    maxVec0 = Vector.Max(maxVec0, Vector.Max(maxVec1, Vector.Max(maxVec2, maxVec3)));
                    VectorHelper.ReduceMinMax(minVec0, maxVec0, ref tmpMin, ref tmpMax);
                }

                while (arr < end)
                {
                    if (*arr < tmpMin) tmpMin = *arr;
                    if (*arr > tmpMax) tmpMax = *arr;
                    arr++;
                }

                min = tmpMin;
                max = tmpMax;
            }
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, ref Vector<double> minVec, ref Vector<double> maxVec, double* end)
            {
#if DEBUG_ASSERT
                Debug.Assert(arr + offset < end);
#endif
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                minVec = Vector.Min(minVec, vec);
                maxVec = Vector.Max(maxVec, vec);
            }
        }
    }
}
