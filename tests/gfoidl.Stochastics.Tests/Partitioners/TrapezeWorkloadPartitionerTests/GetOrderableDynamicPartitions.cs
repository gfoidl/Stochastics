using System;
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
        public void Even_load_size_10_and_2_partitions___returns_2_partitions([Values(0, 1)]double loadFactor)
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
        public void Increasing_load_from_0_size_10_and_2_partitions___returns_2_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 0, 1, 2);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(7, item1.Value.End);
            Assert.AreEqual(7, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(7, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
            Assert.AreEqual(3, item2.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Increasing_load_from_0_size_10_and_3_partitions___returns_3_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 0, 1, 3);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];
            KeyValuePair<long, Range> item3 = actual[2];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);
            TestContext.WriteLine(item3.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(6, item1.Value.End);
            Assert.AreEqual(6, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(6, item2.Value.Start);
            Assert.AreEqual(8, item2.Value.End);
            Assert.AreEqual(2, item2.Value.Size);

            Assert.AreEqual(2, item3.Key);
            Assert.AreEqual(8, item3.Value.Start);
            Assert.AreEqual(10, item3.Value.End);
            Assert.AreEqual(2, item3.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Decreasing_load_to_0_size_10_and_2_partitions___returns_2_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 1, 0, 2);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(3, item1.Value.End);
            Assert.AreEqual(3, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(3, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
            Assert.AreEqual(7, item2.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Decreasing_load_to_0_size_10_and_3_partitions___returns_3_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 1, 0, 3);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

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
            Assert.AreEqual(2, item2.Value.Start);
            Assert.AreEqual(4, item2.Value.End);
            Assert.AreEqual(2, item2.Value.Size);

            Assert.AreEqual(2, item3.Key);
            Assert.AreEqual(4, item3.Value.Start);
            Assert.AreEqual(10, item3.Value.End);
            Assert.AreEqual(6, item3.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Increasing_load_from_5_to_10_size_10_and_2_partitions___returns_2_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 5, 10, 2);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(6, item1.Value.End);
            Assert.AreEqual(6, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(6, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
            Assert.AreEqual(4, item2.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Increasing_load_from_5_to_10_size_10_and_3_partitions___returns_3_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 5, 10, 3);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];
            KeyValuePair<long, Range> item3 = actual[2];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);
            TestContext.WriteLine(item3.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(4, item1.Value.End);
            Assert.AreEqual(4, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(4, item2.Value.Start);
            Assert.AreEqual(7, item2.Value.End);
            Assert.AreEqual(3, item2.Value.Size);

            Assert.AreEqual(2, item3.Key);
            Assert.AreEqual(7, item3.Value.Start);
            Assert.AreEqual(10, item3.Value.End);
            Assert.AreEqual(3, item3.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Decreasing_load_from_10_to_5_size_10_and_2_partitions___returns_2_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 10, 5, 2);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(4, item1.Value.End);
            Assert.AreEqual(4, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(4, item2.Value.Start);
            Assert.AreEqual(10, item2.Value.End);
            Assert.AreEqual(6, item2.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Decreasing_load_from_10_to_5_size_10_and_3_partitions___returns_3_partitions()
        {
            var sut = new TrapezeWorkloadPartitioner(10, 10, 5, 3);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            KeyValuePair<long, Range> item1 = actual[0];
            KeyValuePair<long, Range> item2 = actual[1];
            KeyValuePair<long, Range> item3 = actual[2];

            TestContext.WriteLine(item1.Value);
            TestContext.WriteLine(item2.Value);
            TestContext.WriteLine(item3.Value);

            Assert.AreEqual(0, item1.Key);
            Assert.AreEqual(0, item1.Value.Start);
            Assert.AreEqual(3, item1.Value.End);
            Assert.AreEqual(3, item1.Value.Size);

            Assert.AreEqual(1, item2.Key);
            Assert.AreEqual(3, item2.Value.Start);
            Assert.AreEqual(6, item2.Value.End);
            Assert.AreEqual(3, item2.Value.Size);

            Assert.AreEqual(2, item3.Key);
            Assert.AreEqual(6, item3.Value.Start);
            Assert.AreEqual(10, item3.Value.End);
            Assert.AreEqual(4, item3.Value.Size);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Triangle_huge_size___no_overflow([Values(100_000, 250_000, 1_000_000)] int size)
        {
            var sut = new TrapezeWorkloadPartitioner(size, 0, 1);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            Assert.AreEqual(Environment.ProcessorCount, actual.Count);
            Assert.AreEqual(0, actual[0].Value.Start);
            Assert.Less(actual[0].Value.Start, actual[0].Value.End);

            for (int i = 1; i < actual.Count; ++i)
            {
                Assert.AreEqual(actual[i - 1].Value.End, actual[i].Value.Start);
                Assert.Less(actual[i].Value.Start, actual[i].Value.End);
            }

            Assert.AreEqual(size, actual[actual.Count - 1].Value.End);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Trapeze_huge_size___no_overflow([Values(100_000, 250_000, 1_000_000)] int size)
        {
            var sut = new TrapezeWorkloadPartitioner(size, 1, 1.2);

            List<KeyValuePair<long, Range>> actual = sut.GetOrderableDynamicPartitions().ToList();

            Assert.AreEqual(Environment.ProcessorCount, actual.Count);
            Assert.AreEqual(0, actual[0].Value.Start);
            Assert.Less(actual[0].Value.Start, actual[0].Value.End);

            for (int i = 1; i < actual.Count; ++i)
            {
                Assert.AreEqual(actual[i - 1].Value.End, actual[i].Value.Start);
                Assert.Less(actual[i].Value.Start, actual[i].Value.End);
            }

            Assert.AreEqual(size, actual[actual.Count - 1].Value.End);
        }
    }
}
