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
    public class MinMax
    {
        private readonly int _size;
        //---------------------------------------------------------------------
        public MinMax(int size) => _size = size;
        //---------------------------------------------------------------------
        [Test]
        public void One_value___is_min_and_max()
        {
            double[] values = { 42 };

            var sut = new Sample(values);

            Assert.AreEqual(42, sut.Min);
            Assert.AreEqual(42, sut.Max);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values___correct_MinMax()
        {
            double[] values = { 2, 4, 10, 0 };

            var sut = new Sample(values);

            Assert.AreEqual(0, sut.Min);
            Assert.AreEqual(10, sut.Max);
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

            (double min, double max) actual1 = sut.GetMinMaxSimd();
            (double min, double max) actual2 = sut.GetMinMaxParallelizedSimd();

            Assert.AreEqual(actual1.min, actual2.min, 1e-10);
            Assert.AreEqual(actual1.max, actual2.max, 1e-10);
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

            (double min, double max) actual1 = sut.GetMinMaxSimd();
            double min = values.Min();
            double max = values.Max();

            Assert.AreEqual(min, actual1.min, 1e-10);
            Assert.AreEqual(max, actual1.max, 1e-10);
        }
    }
}