#define PARALLEL
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
    public class SimdAlignedBenchmarksZTransformation : IBenchmark
    {
        public void Run()
        {
#if !DEBUG
            BenchmarkRunner.Run<SimdAlignedBenchmarksZTransformation>();
#endif
        }
        //---------------------------------------------------------------------
        //[Params(1_000, 1_000_000)]
        public int N { get; set; } = 1_000_000;
        //---------------------------------------------------------------------
        public double Mean { get; set; }
        //---------------------------------------------------------------------
        private double[] _values;
        private double   _sigma;
        private Sample   _sample;
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

            _sample   = new Sample(_values);
            _sigma    = _sample.Sigma;
            this.Mean = _sample.Mean;
        }
        //---------------------------------------------------------------------
        [Benchmark(Baseline = true)]
        public double[] Base0() => this.Base(_sigma);
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] Base1() => this.Base(_sigma);
        //---------------------------------------------------------------------
        [Benchmark]
        public double[] Aligned()
        {
            double sigma = _sample.Sigma;
#if PARALLEL
            return _sample.ZTransformationToArrayParallelizedSimd(sigma);
#else
            return _sample.ZTransformationToArraySimd(sigma);
#endif
        }
        //---------------------------------------------------------------------
        internal double[] Base(double sigma)
        {
            var zTrans = new double[this.Count];
#if PARALLEL
            Parallel.ForEach(
                Partitioner.Create(0, _values.Length),
                range => this.Impl_Base(zTrans, sigma, range.Item1, range.Item2)
            );
#else
            this.Impl_Base(zTrans, _sigma, 0, this.N);
#endif

            return zTrans;
        }
        //---------------------------------------------------------------------
        private double ZTransformation(double value, double avg, double sigmaInv) => (value - avg) * sigmaInv;
        //---------------------------------------------------------------------
        private unsafe void Impl_Base(double[] zTrans, double sigma, int i, int n)
        {
            double avg      = this.Mean;
            double sigmaInv = 1d / sigma;

            fixed (double* pSource = _values)
            fixed (double* pTarget = zTrans)
            {
                double* source = pSource + i;
                double* target = pTarget + i;
                n             -= i;
                i              = 0;
                double* end    = source + n;

                if (Vector.IsHardwareAccelerated && (n - i) >= Vector<double>.Count)
                {
                    var avgVec      = new Vector<double>(avg);
                    var sigmaInvVec = new Vector<double>(sigmaInv);

                    // https://github.com/gfoidl/Stochastics/issues/46
                    int m = n & ~(8 * Vector<double>.Count - 1);
                    for (; i < m; i += 8 * Vector<double>.Count)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 1 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 2 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 3 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 4 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 5 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 6 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 7 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 8 * Vector<double>.Count;
                        target += 8 * Vector<double>.Count;
                    }

                    m = n & ~(4 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 1 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 2 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 3 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 4 * Vector<double>.Count;
                        target += 4 * Vector<double>.Count;
                        i      += 4 * Vector<double>.Count;
                    }

                    m = n & ~(2 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);
                        Core(source, target, 1 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 2 * Vector<double>.Count;
                        target += 2 * Vector<double>.Count;
                        i      += 2 * Vector<double>.Count;
                    }

                    m = n & ~(1 * Vector<double>.Count - 1);
                    if (i < m)
                    {
                        Core(source, target, 0 * Vector<double>.Count, avgVec, sigmaInvVec);

                        source += 1 * Vector<double>.Count;
                        target += 1 * Vector<double>.Count;
                    }
                }

                while (source < end)
                {
                    *target = this.ZTransformation(*source, avg, sigmaInv);
                    source++;
                    target++;
                }
            }
            //-----------------------------------------------------------------
            void Core(double* sourceArr, double* targetArr, int offset, Vector<double> avgVec, Vector<double> sigmaInvVec)
            {
                Vector<double> vec       = VectorHelper.GetVector(sourceArr + offset);
                Vector<double> zTransVec = (vec - avgVec) * sigmaInvVec;
                zTransVec.WriteVectorUnaligned(targetArr + offset);
            }
        }
    }
}
