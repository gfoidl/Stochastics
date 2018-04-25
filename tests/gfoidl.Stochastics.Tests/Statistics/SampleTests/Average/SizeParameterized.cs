using System;
using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests.Average
{
    [TestFixture(2)]
    [TestFixture(5)]
    [TestFixture(10)]
    [TestFixture(100)]
    [TestFixture(101)]
    [TestFixture(1_000)]
    [TestFixture(10_000)]
    [TestFixture(100_000)]
    [TestFixture(1_000_000)]
    public class SizeParameterized
    {
        private readonly int _size;
        //---------------------------------------------------------------------
        public SizeParameterized(int size) => _size = size;
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

            Assert.AreEqual(sAvg, pAvg, 1e-7);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Simd_and_Linq_produce_same_result()
        {
            var values = new double[_size];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            sut.CalculateAverageAndVarianceCoreSimd(out double avg, out double variance);
            double actual1 = values.Average();
            double actual2 = avg / _size;

            Assert.AreEqual(actual1, actual2, 1e-10);
        }
    }
}
