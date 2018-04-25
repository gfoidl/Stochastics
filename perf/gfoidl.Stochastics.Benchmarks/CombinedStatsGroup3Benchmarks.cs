//#define SEQUENTIAL
//-----------------------------------------------------------------------------
using System;
using System.Collections.Concurrent;
using System.Numerics;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class CombinedStatsGroup3Benchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs      = new CombinedStatsGroup3Benchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.EachSeparate),align}: {benchs.EachSeparate()}");
            Console.WriteLine($"{nameof(benchs.SimdSequential),align}: {benchs.SimdSequential()}");
            Console.WriteLine($"{nameof(benchs.SimdParallel),align}: {benchs.SimdParallel()}");
#if !DEBUG
            BenchmarkRunner.Run<CombinedStatsGroup3Benchmarks>();
#endif
        }
        //---------------------------------------------------------------------
        [Params(100, 1_000, 10_000, 50_000, 100_000)]
        public int N { get; set; } = 10_000;
        //---------------------------------------------------------------------
        private double[] _values;
        private double   _avg;
        private double   _sigma;
        //---------------------------------------------------------------------
        public int Count                => _values.Length;
        public double Mean              => _avg;
        public double StandardDeviation => _sigma;
        //---------------------------------------------------------------------
        [GlobalSetup]
        public void GlobalSetup()
        {
            _values    = new double[this.N];
            var rnd    = new Random(0);
            double avg = 0;

            for (int i = 0; i < this.N; ++i)
            {
                _values[i] = rnd.NextDouble();
                avg += _values[i];
            }

            _avg = avg / this.Count;

            var sample = new Sample(_values);
            _sigma     = sample.StandardDeviation;
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public (double Kurtosis, double Skewness) EachSeparate()
        {
            var sample = new Sample(_values);

            double kurtosis = sample.Kurtosis;
            double skewness = sample.Skewness;

            return (kurtosis, skewness);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Kurtosis, double Skewness) SimdSequential()
        {
            var (kurtosis, skewness) = this.CombinedImpl((0, this.Count));

            double sigma = this.StandardDeviation;
            double t     = _values.Length * sigma * sigma * sigma;
            kurtosis /= t * sigma;
            skewness /= t;

            return (kurtosis, skewness);
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public (double Kurtosis, double Skewness) SimdParallel()
        {
            double kurtosis = 0;
            double skewness = 0;

            Partitioner<Tuple<int, int>> partitioner = Partitioner.Create(0, _values.Length);
#if SEQUENTIAL
            (double Kurtosis, double Skewness) tmp = this.CombinedImpl((0, this.Count));
            tmp.Kurtosis.SafeAdd(ref kurtosis);
            tmp.Skewness.SafeAdd(ref skewness);
#else
            Parallel.ForEach(
                partitioner,
                range =>
                {
                    (double Kurtosis, double Skewness) tmp = this.CombinedImpl((range.Item1, range.Item2));
                    tmp.Kurtosis.SafeAdd(ref kurtosis);
                    tmp.Skewness.SafeAdd(ref skewness);
                }
            );
#endif
            double sigma = this.StandardDeviation;
            double t     = _values.Length * sigma * sigma * sigma;
            kurtosis /= t * sigma;
            skewness /= t;

            return (kurtosis, skewness);
        }
        //---------------------------------------------------------------------
        private unsafe (double Kurtosis, double Skewness) CombinedImpl((int start, int end) range)
        {
            double kurtosis = 0;
            double skewness = 0;
            double avg      = this.Mean;
            var (i, n)      = range;

            fixed (double* pArray = _values)
            {
                double* arr = pArray + i;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec  = new Vector<double>(avg);
                    var kurtVec = new Vector<double>(0);
                    var skewVec = new Vector<double>(0);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 4 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 5 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 6 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 7 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);

                        arr += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 2 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 3 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);

                        arr += 4 * Vector<double>.Count;
                        i += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);
                        Core(arr, 1 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);

                        arr += 2 * Vector<double>.Count;
                        i += 2 * Vector<double>.Count;
                    }

                    if (i < n - Vector<double>.Count)
                    {
                        Core(arr, 0 * Vector<double>.Count, avgVec, ref kurtVec, ref skewVec);

                        i += Vector<double>.Count;
                    }

                    for (int j = 0; j < Vector<double>.Count; ++j)
                    {
                        kurtosis += kurtVec[j];
                        skewness += skewVec[j];
                    }
                }

                for (; i < n; ++i)
                {
                    double t  = pArray[i] - avg;
                    double t1 = t * t * t;
                    kurtosis += t1 * t;
                    skewness += t1;
                }
            }

            return (kurtosis, skewness);
            //-----------------------------------------------------------------
            void Core(double* arr, int offset, Vector<double> avgVec, ref Vector<double> kurtVec, ref Vector<double> skewVec)
            {
                Vector<double> vec = VectorHelper.GetVectorUnaligned(arr + offset);
                vec -= avgVec;
                Vector<double> tmp = vec * vec * vec;
                kurtVec += tmp * vec;
                skewVec += tmp;
            }
        }
    }
}
