using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.AccuracyTests
{
    [TestFixture]
    public class Epsilon
    {
        [Test]
        public void Epsilon_plus_1___is_Epsilon()
        {
            double a   = 1d;
            double eps = Accuracy.Epsilon;

            Assert.AreEqual(a, a + eps);
        }
    }
}