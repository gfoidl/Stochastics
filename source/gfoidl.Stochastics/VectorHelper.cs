using System.Numerics;
using System.Runtime.CompilerServices;

namespace gfoidl.Stochastics
{
    internal static class VectorHelper
    {
        public static unsafe Vector<double> GetVector(double* arr)
        {
            return Unsafe.Read<Vector<double>>(arr);
        }
        //---------------------------------------------------------------------
        public static unsafe Vector<double> GetVectorWithAdvance(ref double* arr)
        {
            Vector<double> vec = Unsafe.Read<Vector<double>>(arr);
            arr += Vector<double>.Count;

            return vec;
        }
    }
}