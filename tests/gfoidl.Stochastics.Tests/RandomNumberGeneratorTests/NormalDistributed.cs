using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.RandomNumberGeneratorTests
{
    [TestFixture]
    public class NormalDistributed
    {
        [Test]
        public void Different_values_generated_on_each_call()
        {
            var sut = new RandomNumberGenerator();

            double actual1 = sut.NormalDistributed(0.0, 1.0);
            double actual2 = sut.NormalDistributed(0.0, 1.0);

            Assert.AreNotEqual(actual1, actual2);
        }
    }
}
