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
            var values = new double[1000];
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
            var values = new double[1000];
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
            var values = new double[1000];
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
            var values = new double[1000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] sequential = sut.AutoCorrelationSequential().ToArray();
            double[] simd       = sut.AutoCorrelationSimd()      .ToArray();

            for (int i = 0; i < sequential.Length; ++i)
                Assert.AreEqual(sequential[i], simd[i], 1e-10);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array_Values_given___OK()
        {
            var values = new double[1000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] actual = sut.AutoCorrelationToArray();

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
        public void Array_Values_given_Simd___OK()
        {
            var values = new double[1000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] actual = sut.AutoCorrelationToArraySimd();

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
        public void Array_Values_given_ParallelSimd___OK()
        {
            var values = new double[1000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] actual = sut.AutoCorrelationToArrayParallelSimd();

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
        public void Array_Simd_and_ParallelSimd_version_produce_same_result()
        {
            var values = new double[1000];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] simd     = sut.AutoCorrelationToArraySimd();
            double[] parallel = sut.AutoCorrelationToArrayParallelSimd();

            for (int i = 0; i < parallel.Length; ++i)
                Assert.AreEqual(parallel[i], simd[i], 1e-10);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Array_Simd_and_Simd_version_produce_same_result()
        {
            var values = new double[1000];
            var rnd = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            double[] arraySimd = sut.AutoCorrelationToArraySimd();
            double[] simd      = sut.AutoCorrelationSimd().ToArray();

            for (int i = 0; i < simd.Length; ++i)
                Assert.AreEqual(simd[i], arraySimd[i], 1e-10);
        }
    }
}
