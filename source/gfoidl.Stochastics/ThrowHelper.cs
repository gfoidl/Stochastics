using System;
using System.Runtime.CompilerServices;

namespace gfoidl.Stochastics
{
    internal static class ThrowHelper
    {
        internal static void ThrowArgumentNull(string argName)       => throw CreateArgumentNull(argName);
        internal static void ThrowArgumentOutOfRange(string argName) => throw CreateArgumentOutOfRange(argName);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Exception CreateArgumentNull(string argName) => new ArgumentNullException(argName);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Exception CreateArgumentOutOfRange(string argName) => new ArgumentOutOfRangeException(argName);
    }
}