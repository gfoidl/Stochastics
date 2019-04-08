using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.OutlierDetectionTests
{
    [TestFixture]
    public abstract class Base
    {
        private protected class TestOutlierDetection : OutlierDetection
        {
            public TestOutlierDetection(Sample sample) : base(sample) { }
            protected internal override bool IsOutlier((double Value, double zTransformed) value) => false;
        }
    }
}
