using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests.Average
{
    /*
     * Tests are repeated 2x, to test different alignments in memory.
     * I.e. to test random behaviors -> no flaky tests should be there.
     * Robustness is the key ;-)
     */

    [TestFixture]
    public class Simple
    {
        [Test, Repeat(2)]
        public void One_value___is_the_Average()
        {
            double[] values = { 42 };
            var sut         = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(42, actual);
        }
        //---------------------------------------------------------------------
        [Test, Repeat(2)]
        public void Two_values___correct_Avarage()
        {
            double[] values = { 1, 3 };
            var sut         = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(2, actual);
        }
        //---------------------------------------------------------------------
        [Test, Repeat(2)]
        public void Three_values___correct_Avarage()
        {
            double[] values = { 1, 2, 3 };
            var sut         = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(2, actual);
        }
        //---------------------------------------------------------------------
        [Test, Repeat(2)]
        public void Four_values___correct_Average()
        {
            double[] values = { 2, 3, 10, 1 };
            var sut         = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(4, actual);
        }
        //---------------------------------------------------------------------
        [Test, Repeat(2)]
        public void Five_values___correct_Average()
        {
            double[] values = { 2, 4, 3, 10, 1 };
            var sut         = new Sample(values);

            double actual = sut.Mean;

            Assert.AreEqual(4, actual);
        }
    }
}
