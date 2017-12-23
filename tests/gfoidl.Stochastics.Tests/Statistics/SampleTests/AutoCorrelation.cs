using System;
using System.IO;
using System.Linq;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture]
    public class AutoCorrelation
    {
        [Test]
        public void Values_given___OK()
        {
            var values = new double[10_000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] actual = sut.AutoCorrelation().ToArray();

            using (StreamWriter sw = File.CreateText("auto_corr.csv"))
            {
                for (int i = 0; i < actual.Length; ++i)
                    sw.WriteLine($"{i};{actual[i]}");
            }

            Assert.Greater(actual[0], 0.3);

            for (int i = 1; i < actual.Length; ++i)
                Assert.Less(actual[i], 0.3);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values_given_sequential___OK()
        {
            var values = new double[10_000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] actual = sut.AutoCorrelationSequential().ToArray();

            using (StreamWriter sw = File.CreateText("auto_corr.csv"))
            {
                for (int i = 0; i < actual.Length; ++i)
                    sw.WriteLine($"{i};{actual[i]}");
            }

            Assert.Greater(actual[0], 0.3);

            for (int i = 1; i < actual.Length; ++i)
                Assert.Less(actual[i], 0.3);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Values_given_Simd___OK()
        {
            var values = new double[10_000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] actual = sut.AutoCorrelationSimd().ToArray();

            using (StreamWriter sw = File.CreateText("auto_corr.csv"))
            {
                for (int i = 0; i < actual.Length; ++i)
                    sw.WriteLine($"{i};{actual[i]}");
            }

            Assert.Greater(actual[0], 0.3);

            for (int i = 1; i < actual.Length; ++i)
                Assert.Less(actual[i], 0.3);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Sequential_and_Simd_version_produce_same_result()
        {
            var values = new double[10_000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] sequential = sut.AutoCorrelationSequential().ToArray();
            double[] simd       = sut.AutoCorrelationSimd()      .ToArray();

            for (int i = 0; i < sequential.Length; ++i)
                Assert.AreEqual(sequential[i], simd[i], 1e-10);
        }
    }
}