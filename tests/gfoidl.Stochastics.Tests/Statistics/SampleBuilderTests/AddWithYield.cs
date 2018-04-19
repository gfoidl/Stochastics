using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleBuilderTests
{
    [TestFixture]
    public class AddWithYield
    {
        [Test]
        public void IEnumerable_given___correct_Sample()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(values);

            var sut = new SampleBuilder();

            double[] res  = sut.AddWithYield(values.Select(i => i)).ToArray();
            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(values, res);
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

            double[] res  = sut.AddWithYield(values).ToArray();
            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(values, res);
            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
        //---------------------------------------------------------------------
        [Test]
        public void Add_item_then_AddWithYield___correct_Sample_and_correct_yielding()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(new[] { 42d }.Concat(values));

            var sut = new SampleBuilder();

            sut.Add(42d);
            var res       = sut.AddWithYield(values.Select(i => i)).ToArray();
            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(values, res);
            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToArray(), actual.ToArray());
        }
        //---------------------------------------------------------------------
        [Test]
        public void Add_item_then_AddWithYield_Array___correct_Sample_and_correct_yielding()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(new[] { 42d }.Concat(values));

            var sut = new SampleBuilder();

            sut.Add(42d);
            var res       = sut.AddWithYield(values).ToArray();
            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(values, res);
            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToArray(), actual.ToArray());
        }
    }
}
