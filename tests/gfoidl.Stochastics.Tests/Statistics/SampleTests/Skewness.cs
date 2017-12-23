using System;
using System.Linq;
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
    public class Skewness
    {
        private readonly int _size;
        //---------------------------------------------------------------------
        public Skewness(int size) => _size = size;
        //---------------------------------------------------------------------
        [Test]
        public void Simd_and_ParallelizedSimd_produce_same_result()
        {
            var values = new double[_size];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double actual1 = sut.CalculateSkewnessSimd();
            double actual2 = sut.CalculateSkewnessParallelizedSimd();

            Assert.AreEqual(actual1, actual2, 1e-10);
        }
    }
}