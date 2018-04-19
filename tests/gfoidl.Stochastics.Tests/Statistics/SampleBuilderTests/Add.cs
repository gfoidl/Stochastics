using System;
using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleBuilderTests
{
    [TestFixture]
    public class Add
    {
        [Test]
        public void Value_given___correct_Sample()
        {
            double[] values = { Math.PI, Math.E };
            var expected    = new Sample(values);

            var sut = new SampleBuilder();

            sut.Add(Math.PI);
            sut.Add(Math.E);

            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
        //---------------------------------------------------------------------
        [Test]
        public void IEnumerable_given___correct_Sample()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(values);

            var sut = new SampleBuilder();

            sut.Add(values.Select(i => i));

            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array_given___correct_Sample()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(values);

            var sut = new SampleBuilder();

            sut.Add(values);

            Sample actual = sut.GetSample();

            TestContext.WriteLine(actual);

            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
        //---------------------------------------------------------------------
        [Test]
        public void Add_IEnumerable_then_Add_value___correct_Sample()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(values.Concat(new[] { 42d }));

            var sut = new SampleBuilder();

            sut.Add(values.Select(i => i));
            sut.Add(42d);

            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
        //---------------------------------------------------------------------
        [Test]
        public void Add_Array_then_Add_value___correct_Sample()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(values.Concat(new[] { 42d }));

            var sut = new SampleBuilder();

            sut.Add(values);
            sut.Add(42d);

            Sample actual = sut.GetSample();

            TestContext.WriteLine(actual);

            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
    }
}
