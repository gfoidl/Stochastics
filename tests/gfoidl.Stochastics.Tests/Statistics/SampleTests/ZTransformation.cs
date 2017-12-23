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
    public class ZTransformation
    {
        private int _size;
        //---------------------------------------------------------------------
        public ZTransformation(int size) => _size = size;
        //---------------------------------------------------------------------
        [Test]
        public void Values_given___Mean_and_sigma_of_transformed_set_is_0_and_1()
        {
            double[] values = this.GetSampleValues();

            var sut = new Sample(values);

            double[] transformed = sut.ZTransformation().ToArray();
            var actual           = new Sample(transformed);

            TestContext.WriteLine(sut);
            TestContext.WriteLine(actual);

            Assert.AreEqual(0, actual.Mean, 1e-10);
            Assert.AreEqual(1, actual.SampleStandardDeviation, 1e-10);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array_values_given___Mean_and_sigma_of_transformed_set_is_0_and_1()
        {
            double[] values = this.GetSampleValues();

            var sut = new Sample(values);

            double[] transformed = sut.ZTransformationToArray();
            var actual           = new Sample(transformed);

            TestContext.WriteLine(sut);
            TestContext.WriteLine(actual);

            Assert.AreEqual(0, actual.Mean, 1e-10);
            Assert.AreEqual(1, actual.SampleStandardDeviation, 1e-10);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values_given___inverse_transform_results_in_original_values()
        {
            double[] values = this.GetSampleValues();

            var sut              = new Sample(values);
            double[] transformed = sut.ZTransformation().ToArray();

            double sigma = sut.Sigma;
            double avg   = sut.Mean;

            for (int i = 0; i < values.Length; ++i)
            {
                double actual = transformed[i] * sut.Sigma + avg;

                Assert.AreEqual(values[i], actual, 1e-10);
            }
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array_values_given___inverse_transform_results_in_original_values()
        {
            double[] values = this.GetSampleValues();

            var sut              = new Sample(values);
            double[] transformed = sut.ZTransformationToArray();

            double sigma = sut.Sigma;
            double avg   = sut.Mean;

            for (int i = 0; i < values.Length; ++i)
            {
                double actual = transformed[i] * sut.Sigma + avg;

                Assert.AreEqual(values[i], actual, 1e-10);
            }
        }
        //---------------------------------------------------------------------
        [Test]
        public void Repeated_values_given___original_values_returned()
        {
            double[] values = Enumerable.Repeat(42d, 10).ToArray();

            var sut = new Sample(values);

            double[] transformed = sut.ZTransformation().ToArray();

            CollectionAssert.AreEqual(values, transformed);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array_repeated_values_given___original_values_returned()
        {
            double[] values = Enumerable.Repeat(42d, 10).ToArray();

            var sut = new Sample(values);

            double[] transformed = sut.ZTransformationToArray();

            CollectionAssert.AreEqual(values, transformed);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Simd_and_ParallelizedSimd_produce_same_result()
        {
            double[] value = this.GetSampleValues();

            var sut = new Sample(value);

            double[] actual1 = sut.ZTransformationToArraySimd(sut.Sigma);
            double[] actual2 = sut.ZTransformationToArrayParallelizedSimd(sut.Sigma);

            for (int i = 0; i < actual1.Length; ++i)
                Assert.AreEqual(actual1[i], actual2[i], 1e-10);
        }
        //---------------------------------------------------------------------
        private double[] GetSampleValues()
        {
            var values = new double[_size];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            return values;
        }
    }
}