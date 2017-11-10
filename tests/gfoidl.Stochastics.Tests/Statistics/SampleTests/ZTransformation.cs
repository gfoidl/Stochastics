using System;
using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class ZTransformation
    {
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
        private double[] GetSampleValues()
        {
            var values = new double[1000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            return values;
        }
    }
}