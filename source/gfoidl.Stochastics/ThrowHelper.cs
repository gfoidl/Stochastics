using System;
using System.Runtime.CompilerServices;

namespace gfoidl.Stochastics
{
    internal static class ThrowHelper
    {
#if NETSTANDARD
        public static void ThrowArgumentNull(string argName)       => throw new ArgumentNullException(argName);
        public static void ThrowArgumentOutOfRange(string argName) => throw new ArgumentOutOfRangeException(argName);
        public static void ThrowSampleNotInitialized()             => throw new InvalidOperationException(Strings.Sample_must_be_initialized);
#else
        public static void ThrowArgumentNull(string argName)       => throw CreateArgumentNull(argName);
        public static void ThrowArgumentOutOfRange(string argName) => throw CreateArgumentOutOfRange(argName);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Exception CreateArgumentNull(string argName) => new ArgumentNullException(argName);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Exception CreateArgumentOutOfRange(string argName) => new ArgumentOutOfRangeException(argName);
#endif
    }
}