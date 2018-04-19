using System.Collections.Generic;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class SampleEnumerator
    {
        [Test]
        public void Foreach_over_sample___correct_collection()
        {
            double[] values = { 1, 2, 3 };

            var sut = new Sample(values);

            var actual = new List<double>();

            foreach (double item in sut)
                actual.Add(item);

            CollectionAssert.AreEqual(values, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Reset_and_another_iteration___OK()
        {
            double[] values = { 1, 2, 3 };

            var sut = new Sample(values);

            var actual1    = new List<double>();
            var actual2    = new List<double>();
            var enumerator = sut.GetEnumerator();

            while (enumerator.MoveNext())
                actual1.Add(enumerator.Current);

            enumerator.Reset();

            while (enumerator.MoveNext())
                actual2.Add(enumerator.Current);

            CollectionAssert.AreEqual(actual1, actual2);
        }
    }
}
