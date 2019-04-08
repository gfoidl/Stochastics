using System;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class Indexer
    {
        [Test]
        public void Index_is_negative___throws_ArgumentOutOfRange()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                var sut   = new Sample(new double[] { 1d });
                int index = -1;

                double actual = sut[index];
            });
        }
        //---------------------------------------------------------------------
        [Test]
        public void Index_is_above_sample_size___throws_ArgumentOutOfRange()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
            {
                double[] array = { 1d, 2d };
                var sut        = new Sample(array);

                double actual = sut[2];
            });
        }
        //---------------------------------------------------------------------
        [Test]
        public void Correct_value_returned()
        {
            double[] array = { 1d, 2d };
            var sut        = new Sample(array);

            double actual = sut[1];

            Assert.AreEqual(2d, actual, 1e-6);
        }
    }
}
