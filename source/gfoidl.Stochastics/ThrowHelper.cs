using System;
using System.Diagnostics;
using System.Resources;

namespace gfoidl.Stochastics
{
    internal static class ThrowHelper
    {
        private static readonly Lazy<ResourceManager> s_resources;
        //---------------------------------------------------------------------
        static ThrowHelper()
        {
            string ns   = typeof(ThrowHelper).Namespace;
            s_resources = new Lazy<ResourceManager>(() => new ResourceManager($"{ns}.Strings", typeof(ThrowHelper).Assembly));
        }
        //---------------------------------------------------------------------
        public static void ThrowArgumentNull(ExceptionArgument argument)                                   => throw new ArgumentNullException(GetArgumentName(argument));
        public static void ThrowArgumentOutOfRange(ExceptionArgument argument)                             => throw new ArgumentOutOfRangeException(GetArgumentName(argument));
        public static void ThrowArgumentOutOfRange(ExceptionArgument argument, ExceptionResource resource) => throw new ArgumentOutOfRangeException(GetArgumentName(argument), GetResource(resource));
        //---------------------------------------------------------------------
        private static string GetArgumentName(ExceptionArgument argument)
        {
            Debug.Assert(
                Enum.IsDefined(typeof(ExceptionArgument), argument),
                "The enum value is not defined, please check the 'ExceptionArgument' enum.");

            return argument.ToString();
        }
        //---------------------------------------------------------------------
        private static string GetResource(ExceptionResource ressource)
        {
            Debug.Assert(Enum.IsDefined(typeof(ExceptionResource), ressource),
                $"The enum value is not defined, please check the {nameof(ExceptionResource)} enum.");

            return s_resources.Value.GetString(ressource.ToString());
        }
        //---------------------------------------------------------------------
        public enum ExceptionArgument
        {
            array,
            index,
            range,
            values,
            sampleBuilder,
            lambda,
            offset,
            length
        }
        //---------------------------------------------------------------------
        public enum ExceptionResource
        {
            Value_must_be_greater_than_zero
        }
    }
}
