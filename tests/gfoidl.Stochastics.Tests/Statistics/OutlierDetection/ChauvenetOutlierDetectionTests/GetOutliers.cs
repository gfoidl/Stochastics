using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.ChauvenetOutlierDetectionTests
{
    [TestFixture]
    public class GetOutliers
    {
        [Test]
        public void Sample_with_outlier___outlier_reported()
        {
            double[] values   = { 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 1, 999 };
            double[] expected = { 999 };
            var sample        = new Sample(values);

            var sut = new ChauvenetOutlierDetection(sample);

            var actual = sut.GetOutliers().ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Sample_without_outlier___no_outlier_reported()
        {
            double[] values = { 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 1, 3 };
            var sample      = new Sample(values);

            var sut = new ChauvenetOutlierDetection(sample);

            var actual = sut.GetOutliers().ToArray();

            Assert.AreEqual(0, actual.Length);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Sample_with_repeated_values___no_outlier_reported()
        {
            double[] values = Enumerable.Repeat(0.1, 5).ToArray();
            var sample      = new Sample(values);

            var sut = new ChauvenetOutlierDetection(sample);

            var actual = sut.GetOutliers().ToArray();

            Assert.AreEqual(0, actual.Length);
        }
    }
}
