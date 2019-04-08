using System;
using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.OutlierDetectionTests
{
    [TestFixture]
    public class GetOutliers : Base
    {
        [Test]
        public void TestOutlierDetection___Empty()
        {
            double[] values = { 1, 2, 3 };
            var sample      = new Sample(values);

            OutlierDetection sut = new TestOutlierDetection(sample);

            var actual = sut.GetOutliers().ToArray();

            CollectionAssert.AreEqual(Array.Empty<double>(), actual);
        }
    }
}
