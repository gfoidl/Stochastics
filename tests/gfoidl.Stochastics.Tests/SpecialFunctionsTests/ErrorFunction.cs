using System.Collections.Generic;
using NUnit.Framework;
using static gfoidl.Stochastics.SpecialFunctions;

namespace gfoidl.Stochastics.Tests.SpecialFunctionsTests
{
    [TestFixture]
    public class ErrorFunction
    {
        [Test, TestCaseSource(nameof(TestCases))]
        public void Arg_given___correct_result(double x, double erf)
        {
            double res = Erf(x);

            Assert.AreEqual(erf, res, 1e-6);
        }
        //---------------------------------------------------------------------
        private static IEnumerable<TestCaseData> TestCases()
        {
            // https://de.wikipedia.org/wiki/Fehlerfunktion#Wertetabelle
            yield return new TestCaseData(0.00, 0d);
            yield return new TestCaseData(0.50, 0.5204999);
            yield return new TestCaseData(1.00, 0.8427008);
            yield return new TestCaseData(1.50, 0.9661051);
            yield return new TestCaseData(2.00, 0.9953223);
            yield return new TestCaseData(2.50, 0.9995930);
            yield return new TestCaseData(3.00, 0.9999779);
            yield return new TestCaseData(3.50, 0.9999993);
        }
    }
}