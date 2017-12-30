using System.Runtime.CompilerServices;
using System.Threading;

namespace gfoidl.Stochastics
{
    internal static class ExtensionMethods
    {
        // Gets a snapshot of the value and compare-exchanges value
        // when it is still the same as the snapshot
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double SafeAdd(this double amount, ref double value)
        {
            double tmp = double.NaN;

            do
            {
                tmp = value;
            } while (tmp != Interlocked.CompareExchange(ref value, value + amount, tmp));

            return value;
        }
    }
}