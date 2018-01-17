using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.ChauvenetOutlierDetectionTests
{
    [TestFixture]
    public class GetValuesWithoutOutliers
    {
        [Test]
        public void Sample_with_outlier___outlier_reported()
        {
            double[] values   = { 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 999, 2, 0, 2, 2, -100, 3, 2, 5, 2, 3 };
            double[] expected = values.Where(d => d < 10).ToArray();
            var sample        = new Sample(values);

            var sut = new ChauvenetOutlierDetection(sample);

            var actual = sut.GetValuesWithoutOutliers().ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Sample_without_outlier___no_outlier_reported()
        {
            double[] values = { 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3 };
            var sample      = new Sample(values);

            var sut = new ChauvenetOutlierDetection(sample);

            var actual = sut.GetValuesWithoutOutliers().ToArray();

            Assert.AreEqual(values.Length, actual.Length);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Sample_with_repeated_values___no_outlier_reported()
        {
            double[] values = Enumerable.Repeat(0.6, 5).ToArray();
            var sample      = new Sample(values);

            var sut = new ChauvenetOutlierDetection(sample);

            var actual = sut.GetValuesWithoutOutliers().ToArray();

            CollectionAssert.AreEqual(values, actual);
        }
    }
}