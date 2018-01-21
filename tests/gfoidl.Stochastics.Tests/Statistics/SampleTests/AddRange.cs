using System;
using System.Collections.Generic;
using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class AddRange
    {
        [Test]
        public void Values_is_null___throws_ArgumentNull()
        {
            var sut = new Sample();

            Assert.Throws<ArgumentNullException>(() => sut.AddRange(null));
        }
        //---------------------------------------------------------------------
        [Test]
        public void Empty_values___OK()
        {
            var sut = new Sample();

            sut.AddRange(GetValues());
            //-----------------------------------------------------------------
            IEnumerable<double> GetValues()
            {
                yield break;
            }
        }
        //---------------------------------------------------------------------
        [Test]
        public void AddRange_and_ctor___same_result()
        {
            var rnd    = new Random();
            var values = new double[100];

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var expected = new Sample(values);
            var sut      = new Sample();

            sut.AddRange(values.Select(i => i));

            CollectionAssert.AreEqual(expected.Values, sut.Values, nameof(sut.Values));
            Assert.AreEqual(expected.Mean, sut.Mean, 1e-10, nameof(sut.Mean));
            Assert.AreEqual(expected.Min, sut.Min, 1e-10, nameof(sut.Min));
            Assert.AreEqual(expected.Max, sut.Max, 1e-10, nameof(sut.Max));
        }
    }
}