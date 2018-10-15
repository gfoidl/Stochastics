using System.Reflection;
using System.Runtime.Versioning;

namespace gfoidl.Stochastics
{
    internal static class RuntimeHelper
    {
        private static bool? _isRunningOnDotNetCore;
        //---------------------------------------------------------------------
        public static bool IsRunningOnDotNetCore()
        {
            if (!_isRunningOnDotNetCore.HasValue)
            {
                string frameworkName = Assembly.GetEntryAssembly()?.GetCustomAttribute<TargetFrameworkAttribute>()?.FrameworkName;
                _isRunningOnDotNetCore = frameworkName?.Contains("NETCoreApp") ?? false;
            }

            return _isRunningOnDotNetCore.Value;
        }
    }
}
