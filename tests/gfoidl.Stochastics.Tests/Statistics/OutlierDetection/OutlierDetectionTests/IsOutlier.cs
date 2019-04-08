using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.OutlierDetectionTests
{
    [TestFixture]
    public class IsOutlier
    {
        [Test]
        public void Outlier___true()
        {
            double[] values = { 1, 2, 3 };
            var sample      = new Sample(values);

            ChauvenetOutlierDetection sut = new ChauvenetOutlierDetection(sample);

            bool actual = sut.IsOutlier((double.NaN, 15));

            Assert.IsTrue(actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void No_Outlier___false()
        {
            double[] values = { 1, 2, 3 };
            var sample      = new Sample(values);

            ChauvenetOutlierDetection sut = new ChauvenetOutlierDetection(sample);

            bool actual = sut.IsOutlier((double.NaN, 1));

            Assert.IsFalse(actual);
        }
    }
}
