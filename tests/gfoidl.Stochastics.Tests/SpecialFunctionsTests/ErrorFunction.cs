using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using static gfoidl.Stochastics.SpecialFunctions;

namespace gfoidl.Stochastics.Tests.SpecialFunctionsTests
{
    [TestFixture]
    public class ErrorFunction
    {
        [Test, TestCaseSource(nameof(TestCases))]
        public void Erf_Scalar_given___correct_result(double x, double erf)
        {
            double res = Erf(x);

            Assert.AreEqual(erf, res, 1e-6);
        }
        //---------------------------------------------------------------------
        [Test, TestCaseSource(nameof(TestCases))]
        public void Erfc_Scalar_given___correct_result(double x, double erf)
        {
            double res = Erfc(x);

            Assert.AreEqual(1d - erf, res, 1e-6);
        }
        //---------------------------------------------------------------------
        [Test]
        public void Erf_Vector_given___correct_result()
        {
            (double x, double erf)[] testData = TestData().ToArray();

            var values   = new double[testData.Length];
            var expected     = new double[testData.Length];

            for (int i = 0; i < testData.Length; ++i)
            {
                values[i]   = testData[i].x;
                expected[i] = testData[i].erf;
            }

            double[] results = Erf(values);

            Assert.Multiple(() =>
            {
                for (int i = 0; i < testData.Length; ++i)
                    Assert.AreEqual(expected[i], results[i], 1e-6, "failure at index {0}", i);
            });
        }
        //---------------------------------------------------------------------
        [Test]
        public void Erfc_Vector_given___correct_result()
        {
            (double x, double erf)[] testData = TestData().ToArray();

            var values   = new double[testData.Length];
            var expected     = new double[testData.Length];

            for (int i = 0; i < testData.Length; ++i)
            {
                values[i]   = testData[i].x;
                expected[i] = 1d - testData[i].erf;
            }

            double[] results = Erfc(values);

            Assert.Multiple(() =>
            {
                for (int i = 0; i < testData.Length; ++i)
                    Assert.AreEqual(expected[i], results[i], 1e-6, "failure at index {0}", i);
            });
        }
        //---------------------------------------------------------------------
        private static IEnumerable<(double x, double erf)> TestData()
        {
            // https://de.wikipedia.org/wiki/Fehlerfunktion#Wertetabelle
            yield return (0.00, 0d);
            yield return (0.50, 0.5204999);
            yield return (1.00, 0.8427008);
            yield return (1.50, 0.9661051);
            yield return (2.00, 0.9953223);
            yield return (2.50, 0.9995930);
            yield return (3.00, 0.9999779);
            yield return (3.50, 0.9999993);
        }
        //---------------------------------------------------------------------
        private static IEnumerable<TestCaseData> TestCases() => TestData().Select(td => new TestCaseData(td.x, td.erf));
    }
}
