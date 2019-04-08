using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.OutlierDetectionTests
{
    [TestFixture]
    public class GetValuesWithoutOutliers : Base
    {
        [Test]
        public void TestOutlierDetection___Identity()
        {
            double[] values = { 1, 2, 3 };
            var sample      = new Sample(values);

            OutlierDetection sut = new TestOutlierDetection(sample);

            var actual = sut.GetValuesWithoutOutliers().ToArray();

            CollectionAssert.AreEqual(values, actual);
        }
    }
}
