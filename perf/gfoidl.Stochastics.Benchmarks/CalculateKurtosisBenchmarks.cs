using System;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class CalculateKurtosisBenchmarks
    {
        public static void Run()
        {
            var benchs      = new CalculateKurtosisBenchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.Sequential),align}: {benchs.Sequential()}");
            Console.WriteLine($"{nameof(benchs.UnsafeSimd),align}: {benchs.UnsafeSimd()}");
#if !DEBUG
            BenchmarkRunner.Run<CalculateKurtosisBenchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(100, 1_000, 10_000, 100_000, 1_000_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private double[] _values;
        private double   _avg;
        //---------------------------------------------------------------------
        public double Mean              => _avg;
        public double StandardDeviation => 0.34354; // actual value not relevant in bench
        public int Count                => _values.Length;
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
            double kurtosis = 0;
            double avg      = this.Mean;
            double[] arr    = _values;

            for (int i = 0; i < arr.Length; ++i)
            {
                double t  = arr[i] - avg;
                kurtosis += t * t * t * t;
            }

            double sigma = this.StandardDeviation;
            kurtosis /= this.Count * sigma * sigma * sigma * sigma;

            return kurtosis;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double UnsafeSimd()
        {
            double kurtosis = 0;
            double avg      = this.Mean;
            int n           = _values.Length;

            fixed (double* pArray = _values)
            {
                double* arr = pArray;
                int i       = 0;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count * 2)
                {
                    var avgVec  = new Vector<double>(avg);
                    var kurtVec = new Vector<double>(0);

                    for (; i < n - 2 * Vector<double>.Count; i += 2 * Vector<double>.Count)
                    {
                        Vector<double> vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        vec     -= avgVec;
                        kurtVec += vec * vec * vec * vec;

                        vec = VectorHelper.GetVectorWithAdvance(ref arr);
                        vec     -= avgVec;
                        kurtVec += vec * vec * vec * vec;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                        kurtosis += kurtVec[j];
                }

                for (; i < n; ++i)
                {
                    double t = pArray[i] - avg;
                    kurtosis += t * t * t * t;
                }
            }

            double sigma = this.StandardDeviation;
            kurtosis /= n * sigma * sigma * sigma * sigma;

            return kurtosis;
        }
    }
}