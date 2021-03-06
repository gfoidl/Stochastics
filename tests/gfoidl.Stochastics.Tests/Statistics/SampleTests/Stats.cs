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

            Assert.Multiple(() =>
            {
                // Expected values calculated with gnuplot 5.0 patchlevel 1
                Assert.AreEqual(20                                   , sut.Count);
                Assert.AreEqual(51.9500 , sut.Mean                   , 1e-3, nameof(sut.Mean));
                Assert.AreEqual(217.2718, sut.StandardDeviation      , 1e-3, nameof(sut.StandardDeviation));
                Assert.AreEqual(222.9162, sut.SampleStandardDeviation, 1e-3, nameof(sut.SampleStandardDeviation));
                Assert.AreEqual(4.1293  , sut.Skewness               , 1e-3, nameof(sut.Skewness));
                Assert.AreEqual(18.0514 , sut.Kurtosis               , 1e-3, nameof(sut.Kurtosis));
                Assert.AreEqual(94.7050 , sut.Delta                  , 1e-3, nameof(sut.Delta));
            });
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values_given_with_offset___correct_stats()
        {
            double[] values = { -1, -1, 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999, -1 };

            var sut = new Sample(values, 2, 20);

            TestContext.WriteLine(sut);

            Assert.Multiple(() =>
            {
                // Expected values calculated with gnuplot 5.0 patchlevel 1
                Assert.AreEqual(20                                   , sut.Count);
                Assert.AreEqual(51.9500 , sut.Mean                   , 1e-3, nameof(sut.Mean));
                Assert.AreEqual(217.2718, sut.StandardDeviation      , 1e-3, nameof(sut.StandardDeviation));
                Assert.AreEqual(222.9162, sut.SampleStandardDeviation, 1e-3, nameof(sut.SampleStandardDeviation));
                Assert.AreEqual(4.1293  , sut.Skewness               , 1e-3, nameof(sut.Skewness));
                Assert.AreEqual(18.0514 , sut.Kurtosis               , 1e-3, nameof(sut.Kurtosis));
                Assert.AreEqual(94.7050 , sut.Delta                  , 1e-3, nameof(sut.Delta));
            });
        }
    }
}
