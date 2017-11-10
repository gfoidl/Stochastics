using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class Average
    {
        [Test]
        public void One_value___is_the_Average()
        {
            double[] values = { 42 };

            var sut = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(42, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values___correct_Average()
        {
            double[] values = { 2, 4, 10, 0 };

            var sut = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(4, actual);
        }
    }
}