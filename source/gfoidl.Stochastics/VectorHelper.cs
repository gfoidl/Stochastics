using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace gfoidl.Stochastics
{
    [DebuggerNonUserCode]
    internal static unsafe class VectorHelper
    {
        public static Vector<double> GetVector(double* arr)
        {
            return Unsafe.Read<Vector<double>>(arr);
        }
        //---------------------------------------------------------------------
        public static Vector<double> GetVectorWithAdvance(ref double* arr)
        {
            Vector<double> vec = Unsafe.Read<Vector<double>>(arr);
            arr += Vector<double>.Count;

            return vec;
        }
        //---------------------------------------------------------------------
        public static void WriteVector(this Vector<double> vector, double* arr)
        {
            Unsafe.Write(arr, vector);
        }
        //---------------------------------------------------------------------
        public static void WriteVectorWithAdvance(this Vector<double> vector, ref double* arr)
        {
            Unsafe.Write(arr, vector);
            arr += Vector<double>.Count;
        }
    }
}