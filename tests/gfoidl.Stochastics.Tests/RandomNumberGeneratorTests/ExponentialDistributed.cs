using System;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.RandomNumberGeneratorTests
{
    [TestFixture]
    public class ExponentialDistributed
    {
        [Test]
        public void Lambda_is_0___throws_ArgumentOutOfRange()
        {
            var sut = new RandomNumberGenerator();

            Assert.Throws<ArgumentOutOfRangeException>(() => sut.ExponentialDistributed(0));
        }
        //---------------------------------------------------------------------
        [Test]
        public void Lamba_is_not_0___OK([Values(-1, 1, 10, double.MinValue, double.MaxValue)]double lambda)
        {
            var sut = new RandomNumberGenerator();

            double actual = sut.ExponentialDistributed(lambda);

            TestContext.WriteLine(actual);
        }
    }
}