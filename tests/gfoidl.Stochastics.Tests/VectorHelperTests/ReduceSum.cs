using System;
using System.Numerics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.VectorHelperTests
{
    [TestFixture]
    public class ReduceSum
    {
        [Test, Repeat(10)]
        public void Random_Vector_given___Correct_Sum_Reduction()
        {
            if (Vector.IsHardwareAccelerated)
            {
                var rnd         = new Random();
                var arr         = new double[Vector<double>.Count];
                double expected = 0;

                for (int j = 0; j < arr.Length; ++j)
                {
                    arr[j]    = rnd.NextDouble();
                    expected += arr[j];
                }

                var vector = new Vector<double>(arr);

                double actual = vector.ReduceSum();

                Assert.AreEqual(expected, actual, 1e-10);
            }
            else
            {
                Assert.Ignore("Vector.IsHardwareAccelerated is false");
            }
        }
    }
}
