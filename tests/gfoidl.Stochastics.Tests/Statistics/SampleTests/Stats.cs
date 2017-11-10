using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class Stats
    {
        [Test]
        public void Values_given___correct_stats()
        {
            double[] values = { 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999 };

            var sut = new Sample(values);

            TestContext.WriteLine(sut);

            // Expected values calculated with gnuplot 5.0 patchlevel 1
            Assert.AreEqual(51.9500 , sut.Mean   , 1e-3);
            Assert.AreEqual(217.2718, sut.StandardDeviation      , 1e-3);
            Assert.AreEqual(222.9162, sut.SampleStandardDeviation, 1e-3);
            Assert.AreEqual(4.1293  , sut.Skewness   , 1e-3);
            Assert.AreEqual(18.0514 , sut.Kurtosis   , 1e-3);
            Assert.AreEqual(94.7050 , sut.Delta      , 1e-3);
        }
    }
}