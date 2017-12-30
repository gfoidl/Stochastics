using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    [DisassemblyDiagnoser(printSource: true)]
    public class AutoCorrelationBenchmarks
    {
        public static void Run()
        {
            var benchs      = new AutoCorrelationBenchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.SequentialIEnumerable),align}: {GetString(benchs.SequentialIEnumerable())}");
            Console.WriteLine($"{nameof(benchs.Sequential),align}: {GetString(benchs.Sequential())}");
            Console.WriteLine($"{nameof(benchs.UnsafeSequential),align}: {GetString(benchs.UnsafeSequential())}");
            Console.WriteLine($"{nameof(benchs.UnsafeSimd),align}: {GetString(benchs.UnsafeSimd())}");
            Console.WriteLine($"{nameof(benchs.UnsafeParallelSimd),align}: {GetString(benchs.UnsafeParallelSimd())}");
#if !DEBUG
            BenchmarkRunner.Run<AutoCorrelationBenchmarks>();
#endif
            //-----------------------------------------------------------------
            string GetString(double[] array)
            {
                var sb = new StringBuilder();
                int n  = Math.Min(10, array.Length);

                for (int i = 0; i < n; ++i)
                {
                    sb.Append(array[i]);

                    if (i < n - 1) sb.Append(", ");
                }
                sb.Append("  N: ").Append(array.Length);

                return sb.ToString();
            }
        }
        //---------------------------------------------------------------------
        [Params(100, 1_000, 10_000, 100_000)]
        public int N { get; set; } = 1_000;
        //---------------------------------------------------------------------
        private double[] _values;
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
        //[Benchmark(Baseline = true)]
        public double[] SequentialIEnumerable() => this.SequentialImpl(_values).ToArray();
        //---------------------------------------------------------------------
        private IEnumerable<double> SequentialImpl(double[] array)
        {
            int n2 = array.Length >> 1;

            for (int m = 0; m < n2; ++m)
            {
                double r_xx = 0;

                for (int k = m; k < array.Length; ++k)
                    r_xx += array[k] * array[k - m];

                yield return r_xx / (array.Length - m);
            }
        }
        //---------------------------------------------------------------------
        //[Benchmark(Baseline = true)]
        public double[] Sequential()
        {
            double[] array = _values;
            int n2         = array.Length >> 1;
            var corr       = new double[n2];

            for (int m = 0; m < n2; ++m)
            {
                double r_xx = 0;

                // Makes it a bit better, but doesn't elide all bound checks
                if ((uint)m >= (uint)array.Length) this.ThrowIndexOutOfRange();

                for (int k = m; k < array.Length; ++k)
                    r_xx += array[k] * array[k - m];

                corr[m] = r_xx / (array.Length - m);
            }

            return corr;
        }
        //---------------------------------------------------------------------
        private void ThrowIndexOutOfRange() => throw new ArgumentOutOfRangeException();
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public unsafe double[] UnsafeSequential()
        {
            int n    = _values.Length;
            int n2   = n >> 1;
            var corr = new double[n2];

            fixed (double* pArray = _values)
            fixed (double* pCorr  = corr)
            {
                for (int m = 0; m < n2; ++m)
                {
                    double r_xx = 0;

                    for (int k = m; k < n; ++k)
                        r_xx += pArray[k] * pArray[k - m];

                    pCorr[m] = r_xx / (n - m);
                }
            }

            return corr;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double[] UnsafeSimd()
        {
            int n    = _values.Length;
            int n2   = n >> 1;
            var corr = new double[n2];

            this.AutoCorrelationToArrayImpl(corr, (0, n2));

            return corr;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public unsafe double[] UnsafeParallelSimd()
        {
            var corr = new double[_values.Length / 2];

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length / 2),
                range => this.AutoCorrelationToArrayImpl(corr, (range.Item1, range.Item2))
            );

            return corr;
        }
        //---------------------------------------------------------------------
        private unsafe void AutoCorrelationToArrayImpl(double[] corr, (int Start, int End) range)
        {
            int n = _values.Length;

            fixed (double* pArray = _values)
            fixed (double* pCorr = corr)
            {
                for (int m = range.Start; m < range.End; ++m)
                {
                    double r_xx = 0;
                    int k       = m;

                    if (Vector.IsHardwareAccelerated && (n - m) >= Vector<double>.Count * 2)
                    {
                        double* a_k  = &pArray[k];
                        double* a_km = pArray;

                        for (; k < n - 2 * Vector<double>.Count; k += 2 * Vector<double>.Count)
                        {
                            Vector<double> kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            Vector<double> kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);

                            kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);
                        }
                    }

                    for (; k < n; ++k)
                        r_xx += pArray[k] * pArray[k - m];

                    pCorr[m] = r_xx / (n - m);
                }
            }
        }
    }
}