using System;
using System.Numerics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.VectorHelperTests
{
    [TestFixture]
    public class ReduceMinMax
    {
        [Test, Repeat(10)]
        public void Min_Max_Vectors_given___correct_Min_Max()
        {
            var rnd    = new Random();
            var minArr = new double[Vector<double>.Count];
            var maxArr = new double[Vector<double>.Count];
            double min = double.MaxValue;
            double max = double.MinValue;

            for (int j = 0; j < Vector<double>.Count; ++j)
            {
                minArr[j] = rnd.NextDouble();
                maxArr[j] = rnd.NextDouble();

                if (minArr[j] < min) min = minArr[j];
                if (maxArr[j] > max) max = maxArr[j];
            }

            double expectedMin = min;
            double expectedMax = max;

            var minVec = new Vector<double>(minArr);
            var maxVec = new Vector<double>(maxArr);

            VectorHelper.ReduceMinMax(minVec, maxVec, ref min, ref max);

            Assert.Multiple(() =>
            {
                Assert.AreEqual(expectedMin, min, 1e-10);
                Assert.AreEqual(expectedMax, max, 1e-10);
            });
        }
    }
}
