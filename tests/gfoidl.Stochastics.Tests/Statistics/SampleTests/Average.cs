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
    public class Average
    {
        private readonly int _size;
        //---------------------------------------------------------------------
        public Average(int size) => _size = size;
        //---------------------------------------------------------------------
        [Test]
        public void One_value___is_the_Average()
        {
            double[] values = { 42 };

            var sut = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(42, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values___correct_Average()
        {
            double[] values = { 2, 4, 10, 0 };

            var sut = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(4, actual);
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

            double actual1 = sut.CalculateAverageAndVarianceCoreSimd()            .avg;
            double actual2 = sut.CalculateAverageAndVarianceCoreParallelizedSimd().avg;

            Assert.AreEqual(actual1, actual2, 1e-7);
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

            double actual1 = values.Average();
            double actual2 = sut.CalculateAverageAndVarianceCoreSimd().avg / _size;

            Assert.AreEqual(actual1, actual2, 1e-10);
        }
    }
}