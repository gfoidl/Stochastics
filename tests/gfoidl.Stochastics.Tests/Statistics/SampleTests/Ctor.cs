using System;
using System.Collections.Generic;
using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture(typeof(double[]))]
    [TestFixture(typeof(List<double>))]
    [TestFixture(typeof(IEnumerable<double>))]
    public class Ctor<TList> where TList : class, IEnumerable<double>
    {
        private readonly TList    _values;
        private readonly double[] _rawValues;
        //---------------------------------------------------------------------
        public Ctor()
        {
            double[] tmp = { 3, 1, 2 };
            _rawValues   = tmp;

            if (typeof(TList) == typeof(double[]))
                _values = tmp as TList;
            else if (typeof(TList) == typeof(List<double>))
                _values = tmp.ToList() as TList;
            else if (typeof(TList) == typeof(IEnumerable<double>))
                _values = tmp.Select(d => d) as TList;
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values_is_null___throws_ArgumentNull()
        {
            IEnumerable<double> values = null;

            Assert.Throws<ArgumentNullException>(() => new Sample(values));
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values_given___correct_Count()
        {
            var sut = new Sample(_values);

            Assert.AreEqual(3, sut.Count);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values_given___corrent_Values()
        {
            var sut = new Sample(_values);

            IEnumerable<double> actual = sut.Values;

            CollectionAssert.AreEqual(_rawValues, actual);
        }
    }
}
