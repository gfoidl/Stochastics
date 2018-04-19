using System.Collections.Generic;
using System.Linq;
using gfoidl.Stochastics.Builders;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Builders.ArrayBuilderTests
{
    [TestFixture]
    public class AddRange
    {
        [Test]
        public void IEnumerable___correct_result()
        {
            int[] expected = Getvalues().ToArray();
            var sut        = new ArrayBuilder<int>(true);

            sut.AddRange(Getvalues());

            int[] actual = sut.ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array___correct_result()
        {
            int[] expected = Getvalues().ToArray();
            var sut        = new ArrayBuilder<int>(true);

            sut.AddRange(Getvalues().ToArray());

            int[] actual = sut.ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array_with_size_less_than_StartingCapacity_of_ArrayBuilder___OK()
        {
            int[] expected = { 1, 2, 3 };
            var sut        = new ArrayBuilder<int>(true);

            sut.AddRange(expected);

            int[] actual = sut.ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Add_item_then_AddRange_with_array___correct_result()
        {
            int[] expected = new int[] { 42 }.Concat(Getvalues()).ToArray();
            var sut        = new ArrayBuilder<int>(true);

            sut.Add(42);
            sut.AddRange(Getvalues().ToArray());

            int[] actual = sut.ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Add_item_then_AddRange_with_array_then_add_another_item___correct_result()
        {
            int[] expected = new int[] { 42 }.Concat(Getvalues()).Concat(new[] { 3 }).ToArray();
            var sut        = new ArrayBuilder<int>(true);

            sut.Add(42);
            sut.AddRange(Getvalues().ToArray());
            sut.Add(3);

            int[] actual = sut.ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Add_item_up_to_StartingCapacity_then_AddRange_with_array_then_add_another_item___correct_result()
        {
            int[] expected = new int[] { 42, 41, 40, 39 }.Concat(Getvalues()).Concat(new[] { 3 }).ToArray();
            var sut = new ArrayBuilder<int>(true);

            sut.Add(42);
            sut.Add(41);
            sut.Add(40);
            sut.Add(39);
            sut.AddRange(Getvalues().ToArray());
            sut.Add(3);

            int[] actual = sut.ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        [Test]
        public void AddRange_with_array_then_Add_item___correct_result()
        {
            int[] expected = Getvalues().Concat(new[] { 42 }).ToArray();
            var sut        = new ArrayBuilder<int>(true);

            sut.AddRange(Getvalues().ToArray());
            sut.Add(42);

            int[] actual = sut.ToArray();

            CollectionAssert.AreEqual(expected, actual);
        }
        //---------------------------------------------------------------------
        private static IEnumerable<int> Getvalues()
        {
            yield return 0;
            yield return 1;
            yield return 2;
            yield return 4;
            yield return 8;
            yield return 16;
            yield return 100;
            yield return 1_000;
            yield return 1_001;
        }
    }
}
