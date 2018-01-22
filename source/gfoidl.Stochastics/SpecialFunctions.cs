using static System.Math;

namespace gfoidl.Stochastics
{
    /// <summary>
    /// Provides methods for special functions.
    /// </summary>
    public static class SpecialFunctions
    {
        /// <summary>
        /// Returns the value of the gaussian error function at <paramref name="x" />.
        /// </summary>
        /// <param name="x">The argument.</param>
        /// <returns>The value of the gaussian error function by <paramref name="x" />.</returns>
        /// <seealso cref="!:https://www.johndcook.com/blog/csharp_erf/" />
        public static double Erf(double x)
        {
            const double a1 = 0.254829592;
            const double a2 = -0.284496736;
            const double a3 = 1.421413741;
            const double a4 = -1.453152027;
            const double a5 = 1.061405429;
            const double p  = 0.3275911;

            // Save the sign of x
            int sign = Sign(x);
            x        = Abs(x);

            // A&S formula 7.1.26
            double t = 1.0 / (1.0 + p * x);
            double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Exp(-x * x);

            return sign * y;
        }
    }
}