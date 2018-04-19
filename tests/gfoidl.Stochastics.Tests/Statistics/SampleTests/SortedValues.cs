using System.Collections.Generic;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class SortedValues
    {
        [Test]
        public void Values_given___correct_sort_order()
        {
            double[] values   = { 3, 1, 2 };
            double[] expected = { 1, 2, 3 };

            var sut = new Sample(values);

            ICollection<double> actual = sut.SortedValues;

            CollectionAssert.AreEqual(expected, actual);
        }
    }
}
