using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture(10)]
    [TestFixture(100)]
    [TestFixture(1_000)]
    [TestFixture(10_000)]
    [TestFixture(100_000)]
    [TestFixture(1_000_000)]
    public class Delta
    {
        private readonly int _size;
        //---------------------------------------------------------------------
        public Delta(int size) => _size = size;
        //---------------------------------------------------------------------
        [Test]
        public void Same_value___delta_is_0()
        {
            var values = new double[_size];

            var sut = new Sample(values);

            double actual = sut.Delta;

            Assert.AreEqual(0, actual, 1e-10);
        }
    }
}