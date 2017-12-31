using System.Collections.Generic;
using System.Linq;
using gfoidl.Stochastics.Partitioners;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Partitioners.StaticRangePartitionerTests
{
    [TestFixture]
    public class GetOrderableDynamicPartitions
    {
        [Test]
        public void Size_10_and_2_partitions___returns_2_partitions()
        {
            var sut = new StaticRangePartitioner(10, 2);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(5, item1.Value.End);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(5, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Size_10_and_3_partitions___returns_4_partitions()
        {
            var sut = new StaticRangePartitioner(10, 3);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];
            KeyValuePair<long, Range> item3 = actual[2];
            KeyValuePair<long, Range> item4 = actual[3];

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(3, item1.Value.End);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(3, item2.Value.Start);
            Assert.AreEqual(6, item2.Value.End);

            Assert.AreEqual(2, item3.Key);
            Assert.AreEqual(6, item3.Value.Start);
            Assert.AreEqual(9, item3.Value.End);

            Assert.AreEqual(3, item4.Key);
            Assert.AreEqual(9, item4.Value.Start);
            Assert.AreEqual(10, item4.Value.End);
        }
    }
}