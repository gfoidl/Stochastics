using System.Collections.Generic;
using System.Linq;
using gfoidl.Stochastics.Partitioners;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Partitioners.TrapezeWorkloadPartitionerTests
{
    [TestFixture]
    public class GetOrderableDynamicPartitions
    {
        [Test]
        public void Even_load_size_10_and_2_partitions___returns_2_equal_sized_partitions([Values(0, 1)]double loadFactor)
        {
            var sut = new TrapezeWorkloadPartitioner(10, loadFactor, loadFactor, 2);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(5, item1.Value.End);
            Assert.AreEqual(5, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(5, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
            Assert.AreEqual(5, item2.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Increasing_load_from_0_size_10_and_2_partitions___returns_2_partitions_of_equal_work()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 0, 1, 2);

            var actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(7, item1.Value.End);
            Assert.AreEqual(7, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(item1.Value.End, item2.Value.Start);
            Assert.Less(item2.Value.Size, item1.Value.Size);
            Assert.AreEqual(7, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
            Assert.AreEqual(3, item2.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Increasing_load_from_0_size_10_and_3_partitions___returns_3_partitions_of_equal_work()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 0, 1, 3);

            var actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];
            KeyValuePair<long, Range> item3 = actual[2];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);
            TestContext.WriteLine(item3.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(5, item1.Value.End);
            Assert.AreEqual(5, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(item1.Value.End, item2.Value.Start);
            Assert.Less(item2.Value.Size, item1.Value.Size);
            Assert.AreEqual(5, item2.Value.Start);
            Assert.AreEqual(8, item2.Value.End);
            Assert.AreEqual(3, item2.Value.Size);

            Assert.AreEqual(2, item3.Key);
            Assert.AreEqual(item2.Value.End, item3.Value.Start);
            Assert.Less(item3.Value.Size, item2.Value.Size);
            Assert.AreEqual(8, item3.Value.Start);
            Assert.AreEqual(10, item3.Value.End);
            Assert.AreEqual(2, item3.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Decreasing_load_to_0_size_10_and_2_partitions___returns_2_partitions_of_equal_work()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 1, 0, 2);

            var actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(3, item1.Value.End);
            Assert.AreEqual(3, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(item1.Value.End, item2.Value.Start);
            Assert.Less(item1.Value.Size, item2.Value.Size);
            Assert.AreEqual(3, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
            Assert.AreEqual(7, item2.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Decreasing_load_to_0_size_10_and_3_partitions___returns_3_partitions_of_equal_work()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 1, 0, 3);

            var actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];
            KeyValuePair<long, Range> item3 = actual[2];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);
            TestContext.WriteLine(item3.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(2, item1.Value.End);
            Assert.AreEqual(2, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(item1.Value.End, item2.Value.Start);
            Assert.Less(item1.Value.Size, item2.Value.Size);
            Assert.AreEqual(2, item2.Value.Start);
            Assert.AreEqual(5, item2.Value.End);
            Assert.AreEqual(3, item2.Value.Size);

            Assert.AreEqual(2, item3.Key);
            Assert.AreEqual(item2.Value.End, item3.Value.Start);
            Assert.Less(item2.Value.Size, item3.Value.Size);
            Assert.AreEqual(5, item3.Value.Start);
            Assert.AreEqual(10, item3.Value.End);
            Assert.AreEqual(5, item3.Value.Size);
        }
    }
}