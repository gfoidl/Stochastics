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
            double snapshot = double.NaN;

            do
            {
                snapshot = value;
            } while (snapshot != Interlocked.CompareExchange(ref value, value + amount, snapshot));

            return value;
        }
        //---------------------------------------------------------------------
        public static bool InterlockedExchangeIfGreater(this double comparison, ref double location)
            => InterlockedExchangeIfGreater(comparison, ref location, comparison);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool InterlockedExchangeIfGreater(this double comparison, ref double location, double newValue)
        {
            double snapshot = double.NaN;

            do
            {
                snapshot = location;

                if (snapshot > comparison) return false;
            } while (snapshot != Interlocked.CompareExchange(ref location, newValue, snapshot));

            return true;
        }
        //---------------------------------------------------------------------
        public static bool InterlockedExchangeIfSmaller(this double comparison, ref double location)
            => InterlockedExchangeIfSmaller(comparison, ref location, comparison);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool InterlockedExchangeIfSmaller(this double comparison, ref double location, double newValue)
        {
            double snapshot = double.NaN;

            do
            {
                snapshot = location;

                if (snapshot < comparison) return false;
            } while (snapshot != Interlocked.CompareExchange(ref location, newValue, snapshot));

            return true;
        }
    }
}