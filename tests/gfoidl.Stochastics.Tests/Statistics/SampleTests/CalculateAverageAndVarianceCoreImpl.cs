using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class CalculateAverageAndVarianceCoreImpl
    {
        [Test]
        public void Values_with_offset___OK()
        {
            double[] values = { -1, -1, 2, 3, 10, 1, -1 };
            var sut         = new Sample(values);

            sut.CalculateAverageAndVarianceCoreImpl(2, 6, out double avg, out double var);

            Assert.AreEqual(4 * 4, avg);        // avg isn't divided by length
        }
    }
}
