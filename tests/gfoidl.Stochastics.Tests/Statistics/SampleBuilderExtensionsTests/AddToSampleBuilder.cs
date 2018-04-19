using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleBuilderExtensionsTests
{
    [TestFixture]
    public class AddToSampleBuilder
    {
        [Test]
        public void Values_given_to_extension_method___correct_Sample()
        {
            double[] values = { 1, 2, 3 };
            var expected    = new Sample(values);

            var sut = new SampleBuilder();

            values.AddToSampleBuilder(sut).ToList();

            Sample actual = sut.GetSample();

            CollectionAssert.AreEqual(expected.Values, actual.Values);
            Assert.AreEqual(expected.ToString(), actual.ToString());
        }
    }
}
