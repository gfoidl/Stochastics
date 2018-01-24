using System.Collections.Generic;
using System.Numerics;
using gfoidl.Stochastics.Builders;
using static System.Math;
using static gfoidl.Stochastics.MathConstants;
using static gfoidl.Stochastics.SpecialFunctions;

namespace gfoidl.Stochastics.Statistics
{
    /// <summary>
    /// Outlier detection based on Chauvenet's criterion.
    /// </summary>
    public class ChauvenetOutlierDetection : OutlierDetection
    {
        private double[] _tsusSqrt2Inv;
        //---------------------------------------------------------------------
        /// <summary>
        /// Creates a new instance of <see cref="ChauvenetOutlierDetection" />
        /// </summary>
        /// <param name="sample">
        /// The <see cref="Statistics.Sample" /> on that outlier detection
        /// is performed.
        /// </param>
        /// <exception cref="System.ArgumentNullException">
        /// <paramref name="sample" /> is <c>null</c>.
        /// </exception>
        public ChauvenetOutlierDetection(Sample sample) : base(sample) { }
        //---------------------------------------------------------------------
        /// <summary>
        /// Determines if the <paramref name="value" /> is an outlier or not.
        /// </summary>
        /// <param name="value">The value to check for outlier.</param>
        /// <returns>
        /// <c>true</c> if the value is an outlier, <c>false</c> if the value
        /// is not an outlier.
        /// </returns>
        protected override bool IsOutlier((double Value, double zTransformed) value)
        {
            double tsus = Abs(value.zTransformed);
            //double probOutside = 1 - Erf(tsus * _sqrt2Inv);
            // Erfc = 1 - Erf -> so use this :-)
            double probOutside = Erfc(tsus * Sqrt2Inv);

            return IsOutlier(probOutside, this.Sample.Count);
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Gets the outliers of the <see cref="P:gfoidl.Stochastics.Statistics.OutlierDetection.Sample" />.
        /// </summary>
        /// <returns>The outliers of the <see cref="P:gfoidl.Stochastics.Statistics.OutlierDetection.Sample" />.</returns>
        public override IEnumerable<double> GetOutliers()
        {
            double[] sample           = this.Sample.Values;
            double[] tsusSqrt2Inv = this.GetTsusSqrt2Inv();
            double[] probOutside  = Erfc(tsusSqrt2Inv);

            var arrayBuilder = new ArrayBuilder<double>(true);
            int n            = this.Sample.Count;

            for (int i = 0; i < probOutside.Length; ++i)
            {
                if (IsOutlier(probOutside[i], n))
                    arrayBuilder.Add(sample[i]);
            }

            return arrayBuilder.ToEnumerable();
        }
        //---------------------------------------------------------------------
        /// <summary>
        /// Gets the values of the <see cref="P:gfoidl.Stochastics.Statistics.OutlierDetection.Sample" /> without outliers.
        /// </summary>
        /// <returns>The values of the <see cref="P:gfoidl.Stochastics.Statistics.OutlierDetection.Sample" /> without outliers.</returns>
        public override IEnumerable<double> GetValuesWithoutOutliers()
        {
            double[] sample           = this.Sample.Values;
            double[] tsusSqrt2Inv = this.GetTsusSqrt2Inv();
            double[] probOutside  = Erfc(tsusSqrt2Inv);

            var arrayBuilder = new ArrayBuilder<double>(true);
            int n            = this.Sample.Count;

            for (int i = 0; i < probOutside.Length; ++i)
            {
                if (!IsOutlier(probOutside[i], n))
                    arrayBuilder.Add(sample[i]);
            }

            return arrayBuilder.ToEnumerable();
        }
        //---------------------------------------------------------------------
        private static bool IsOutlier(double probOutside, double n) => n * probOutside < 0.5;
        //---------------------------------------------------------------------
        private double[] GetTsusSqrt2Inv()
        {
            return _tsusSqrt2Inv ?? (_tsusSqrt2Inv = this.GetTsusSqrt2InvCore());
        }
        //---------------------------------------------------------------------
        private unsafe double[] GetTsusSqrt2InvCore()
        {
            double[] zTransformed = this.Sample.ZTransformationToArray();
            double[] tsusSqrt2Inv = new double[zTransformed.Length];

            fixed (double* pSource = zTransformed)
            fixed (double* pTarget = tsusSqrt2Inv)
            {
                double* zTrans = pSource;
                double* tsus   = pTarget;
                int i          = 0;
                int n          = zTransformed.Length;

                if (Vector.IsHardwareAccelerated && n >= Vector<double>.Count)
                {
                    var sqrt2Inv = new Vector<double>(Sqrt2Inv);

                    for (; i < n - 8 * Vector<double>.Count; i += 8 * Vector<double>.Count)
                    {
                        Core(zTrans, tsus, 0 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 1 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 2 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 3 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 4 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 5 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 6 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 7 * Vector<double>.Count, sqrt2Inv);

                        zTrans += 8 * Vector<double>.Count;
                        tsus   += 8 * Vector<double>.Count;
                    }

                    if (i < n - 4 * Vector<double>.Count)
                    {
                        Core(zTrans, tsus, 0 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 1 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 2 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 3 * Vector<double>.Count, sqrt2Inv);

                        zTrans += 4 * Vector<double>.Count;
                        tsus   += 4 * Vector<double>.Count;
                        i      += 4 * Vector<double>.Count;
                    }

                    if (i < n - 2 * Vector<double>.Count)
                    {
                        Core(zTrans, tsus, 0 * Vector<double>.Count, sqrt2Inv);
                        Core(zTrans, tsus, 1 * Vector<double>.Count, sqrt2Inv);

                        zTrans += 2 * Vector<double>.Count;
                        tsus   += 2 * Vector<double>.Count;
                        i      += 2 * Vector<double>.Count;
                    }

                    if (i < n * Vector<double>.Count)
                    {
                        Core(zTrans, tsus, 0 * Vector<double>.Count, sqrt2Inv);

                        i += Vector<double>.Count;
                    }
                }

                for (; i < n; ++i)
                    tsus[i] = Abs(zTrans[i]) * Sqrt2Inv;
            }

            return tsusSqrt2Inv;
            //-----------------------------------------------------------------
            void Core(double* zTrans, double* tsus, int offset, Vector<double> sqrt2Inv)
            {
                Vector<double> t = Vector.Abs(VectorHelper.GetVector(zTrans + offset));
                t *= sqrt2Inv;
                t.WriteVector(tsus + offset);
            }
        }
    }
}
