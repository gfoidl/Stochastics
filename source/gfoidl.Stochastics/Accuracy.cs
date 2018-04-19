//#define ITERATIVE
//-----------------------------------------------------------------------------
#if !ITERATIVE
// https://en.wikipedia.org/wiki/Machine_epsilon
using System.Runtime.InteropServices;
#endif

namespace gfoidl.Stochastics
{
    internal static class Accuracy
    {
        private static double _epsilon = double.NaN;
        //---------------------------------------------------------------------
        internal static double Epsilon
        {
            get
            {
                if (double.IsNaN(_epsilon))
                {
#if ITERATIVE
                    double tau  = 1;
                    double walt = 1;
                    double wneu = 0;

                    while (wneu != walt)
                    {
                        tau *= 0.5;
                        wneu = walt + tau;
                    }

                    _epsilon = tau;
#else
                    var s    = new dbl();
                    s.d64    = 1d;
                    s.i64++;
                    _epsilon = (s.d64 - 1d) * 0.5;
#endif
                }

                return _epsilon;
            }
        }
        //---------------------------------------------------------------------
#if !ITERATIVE
        [StructLayout(LayoutKind.Explicit)]
        private struct dbl
        {
            [FieldOffset(0)]
            public ulong i64;

            [FieldOffset(0)]
            public double d64;
        }
#endif
    }
}
