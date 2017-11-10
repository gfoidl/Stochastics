using System;

namespace gfoidl.Stochastics
{
    /// <summary>
    /// A pseudo-random number generator, that produces samples in the
    /// range [0, 1).
    /// </summary>
    public class RandomNumberGenerator
    {
        private double _uz;
        private double _nz;
        private double _ez;
        //---------------------------------------------------------------------
        /// <summary>
        /// Initializes the <see cref="RandomNumberGenerator" /> with default
        /// values.
        /// </summary>
        /// <remarks>
        /// Calls <see cref="RandomNumberGenerator(double)" /> with
        /// argument 0.23906
        /// </remarks>
        public RandomNumberGenerator() : this(0.23906) { }
        //---------------------------------------------------------------------
        /// <summary>
        /// Initializes the <see cref="RandomNumberGenerator" />.
        /// </summary>
        /// <param name="initGZ">Parameter for uniform distributes values</param>
        public RandomNumberGenerator(double initGZ)
        {
            _uz = initGZ;
            _nz = 0d;
            _ez = 0d;
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Produces the next uniform distributes number in the range [0, 1).
        /// </summary>
        /// <returns>The next uniform distributes number in the range [0, 1).</returns>
        public double Uniform()
        {
            _uz = (201d * _uz + 100000d / 3d) - (int)(201d * _uz + 100000d / 3d);

            return _uz;
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Produces the next normally distributes number.
        /// </summary>
        /// <param name="mu">Center of the normal distribution</param>
        /// <param name="sigma">Sigma (width) of the normal distribution</param>
        /// <returns>The next normally distributes number.</returns>
        public double NormalDistributed(double mu, double sigma)
        {
            _nz =
                sigma * Math.Sqrt(-2d * Math.Log(Uniform(), 2d))
                * Math.Sin(2d * Math.PI * Uniform()) + mu;

            return _nz;
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Produces the next exponentially distributes number.
        /// </summary>
        /// <param name="lambda">Parameter in the exponential distribution</param>
        /// <returns>The next exponentially distributes number.</returns>
        public double ExponentialDistributed(double lambda)
        {
            if (Math.Abs(lambda - 0) < Accuracy.Epsilon)
                throw new ArgumentOutOfRangeException(nameof(lambda), Strings.Value_must_be_greater_than_zero);

            _ez = -(1d / lambda) * Math.Log(Uniform() + 1e-12, 2d);

            return _ez;
        }
    }
}