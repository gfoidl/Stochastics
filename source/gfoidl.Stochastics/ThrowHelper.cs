using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace gfoidl.Stochastics
{
    internal static class ThrowHelper
    {
#if NETSTANDARD
        public static void ThrowArgumentNull(ExceptionArgument argument)       => throw new ArgumentNullException(GetArgumentName(argument));
        public static void ThrowArgumentOutOfRange(ExceptionArgument argument) => throw new ArgumentOutOfRangeException(GetArgumentName(argument));
        public static void ThrowArgumentOutOfRange(ExceptionArgument argument, ExceptionResource resource) => throw new ArgumentOutOfRangeException(GetArgumentName(argument), GetResourceText(resource));
#else
        public static void ThrowArgumentNull(string argName)           => throw CreateArgumentNull(argName);
        public static void ThrowArgumentOutOfRange(string argName) => throw CreateArgumentOutOfRange(argName);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Exception CreateArgumentNull(string argName) => new ArgumentNullException(argName);
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static Exception CreateArgumentOutOfRange(string argName) => new ArgumentOutOfRangeException(argName);
#endif
        //---------------------------------------------------------------------
        private static string GetArgumentName(ExceptionArgument argument)
        {
            Debug.Assert(
                Enum.IsDefined(typeof(ExceptionArgument), argument),
                "The enum value is not defined, please check the 'ExceptionArgument' enum.");

            return argument.ToString();
        }
        //---------------------------------------------------------------------
        private static string GetResourceName(ExceptionResource resource)
        {
            Debug.Assert(
                Enum.IsDefined(typeof(ExceptionResource), resource),
                "The enum value is not defined, please check the 'ExceptionResource' enum.");

            return resource.ToString();
        }
        //---------------------------------------------------------------------
        private static string GetResourceText(ExceptionResource resource)
            => Strings.ResourceManager.GetString(GetResourceName(resource), Strings.Culture);
        //---------------------------------------------------------------------
        public enum ExceptionArgument
        {
            array,
            index,
            range,
            values,
            sampleBuilder,
            lambda
        }
        //---------------------------------------------------------------------
        public enum ExceptionResource
        {
            Value_must_be_greater_than_zero
        }
    }
}
