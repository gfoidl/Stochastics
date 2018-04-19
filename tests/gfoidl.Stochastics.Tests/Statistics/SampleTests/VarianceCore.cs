using System;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture(10)]
    [TestFixture(100)]
    [TestFixture(1_000)]
    [TestFixture(10_000)]
    [TestFixture(100_000)]
    [TestFixture(1_000_000)]
    public class VarianceCore
    {
        private readonly int _size;
        //---------------------------------------------------------------------
        public VarianceCore(int size) => _size = size;
        //---------------------------------------------------------------------
        [Test]
        public void Same_value___delta_is_0()
        {
            var values = new double[_size];

            var sut = new Sample(values);

            double actual = sut.Variance;

            Assert.AreEqual(0, actual, 1e-10);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Simd_and_ParallelizedSimd_produce_same_result()
        {
            var values = new double[_size];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            sut.CalculateAverageAndVarianceCoreSimd(out double sAvg, out double sVariance);
            sut.CalculateAverageAndVarianceCoreParallelizedSimd(out double pAvg, out double pVariance);

            Assert.AreEqual(sVariance, pVariance, 1e-7);
        }
    }
}
