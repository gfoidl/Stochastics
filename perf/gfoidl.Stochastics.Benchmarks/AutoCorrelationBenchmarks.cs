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
    public class AutoCorrelationBenchmarks : IBenchmark
    {
        public void Run()
        {
            var benchs      = new AutoCorrelationBenchmarks();
            benchs.N        = 1000;
            benchs.GlobalSetup();
            const int align = -25;
            Console.WriteLine($"{nameof(benchs.SequentialIEnumerable),align}: {GetString(benchs.SequentialIEnumerable())}");
            Console.WriteLine($"{nameof(benchs.Sequential),align}: {GetString(benchs.Sequential())}");
            Console.WriteLine($"{nameof(benchs.UnsafeSequential),align}: {GetString(benchs.UnsafeSequential())}");
            Console.WriteLine($"{nameof(benchs.UnsafeSimd),align}: {GetString(benchs.UnsafeSimd())}");
            Console.WriteLine($"{nameof(benchs.UnsafeSimdUnrolled),align}: {GetString(benchs.UnsafeSimdUnrolled())}");
            Console.WriteLine($"{nameof(benchs.UnsafeParallelSimd),align}: {GetString(benchs.UnsafeParallelSimd())}");
            Console.WriteLine($"{nameof(benchs.UnsafeParallelSimdUnrolled),align}: {GetString(benchs.UnsafeParallelSimdUnrolled())}");
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
        //[Benchmark(Baseline = true)]
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
        //[Benchmark(Baseline = true)]
        public double[] UnsafeSimd()
        {
            int n    = _values.Length;
            int n2   = n >> 1;
            var corr = new double[n2];

            this.AutoCorrelationToArrayImpl(corr, (0, n2));

            return corr;
        }
        //---------------------------------------------------------------------
        //[Benchmark]
        public double[] UnsafeSimdUnrolled()
        {
            int n    = _values.Length;
            int n2   = n >> 1;
            var corr = new double[n2];

            this.AutoCorrelationToArrayImplUnrolled(corr, (0, n2));

            return corr;
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double[] UnsafeParallelSimd()
        {
            var corr = new double[_values.Length / 2];

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length / 2),
                range => this.AutoCorrelationToArrayImpl(corr, (range.Item1, range.Item2))
            );

            return corr;
        }
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] UnsafeParallelSimdUnrolled()
        {
            var corr = new double[_values.Length / 2];

            Parallel.ForEach(
                Partitioner.Create(0, _values.Length / 2),
                range => this.AutoCorrelationToArrayImplUnrolled(corr, (range.Item1, range.Item2))
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
#pragma warning disable CS0618
                            Vector<double> kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            Vector<double> kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);

                            kVec  = VectorHelper.GetVectorWithAdvance(ref a_k);
                            kmVec = VectorHelper.GetVectorWithAdvance(ref a_km);
                            r_xx += Vector.Dot(kVec, kmVec);
#pragma warning restore CS0618
                        }
                    }

                    for (; k < n; ++k)
                        r_xx += pArray[k] * pArray[k - m];

                    pCorr[m] = r_xx / (n - m);
                }
            }
        }
        //---------------------------------------------------------------------
        private unsafe void AutoCorrelationToArrayImplUnrolled(double[] corr, (int Start, int End) range)
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

                        for (; k < n - 8 * Vector<double>.Count; k += 8 * Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 1 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 2 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 3 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 4 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 5 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 6 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 7 * Vector<double>.Count, ref r_xx);

                            a_k  += 8 * Vector<double>.Count;
                            a_km += 8 * Vector<double>.Count;
                        }

                        if (k < n - 4 * Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 1 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 2 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 3 * Vector<double>.Count, ref r_xx);

                            a_k  += 4 * Vector<double>.Count;
                            a_km += 4 * Vector<double>.Count;
                            k    += 4 * Vector<double>.Count;
                        }

                        if (k < n - 2 * Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);
                            Core(a_k, a_km, 1 * Vector<double>.Count, ref r_xx);

                            a_k  += 2 * Vector<double>.Count;
                            a_km += 2 * Vector<double>.Count;
                            k    += 2 * Vector<double>.Count;
                        }

                        if (k < n - Vector<double>.Count)
                        {
                            Core(a_k, a_km, 0 * Vector<double>.Count, ref r_xx);

                            k += Vector<double>.Count;
                        }
                    }

                    for (; k < n; ++k)
                        r_xx += pArray[k] * pArray[k - m];

                    pCorr[m] = r_xx / (n - m);
                }
            }
            //-----------------------------------------------------------------
            void Core(double* a_k, double* a_km, int offset, ref double r_xx)
            {
                Vector<double> kVec  = VectorHelper.GetVector(a_k + offset);
                Vector<double> kmVec = VectorHelper.GetVector(a_km + offset);
                r_xx += Vector.Dot(kVec, kmVec);
            }
        }
    }
}
