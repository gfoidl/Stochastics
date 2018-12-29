using System;
using System.Runtime.InteropServices;
using gfoidl.Stochastics.Statistics;

namespace gfoidl.Stochastics.Native
{
    internal static class Gpu
    {
        public const string EnvVariableName = "GFOIDL_STOCHASTICS_USE_GPU";
        //---------------------------------------------------------------------
        private static readonly bool s_isAvailable;
        private static readonly bool s_isUseOfGpuForced;
        //---------------------------------------------------------------------
        static Gpu()
        {
            string env = Environment.GetEnvironmentVariable(EnvVariableName);

            s_isAvailable = RuntimeInformation.OSArchitecture == Architecture.X64
                && RuntimeHelper.IsRunningOnDotNetCore()
                && GpuMethods.gpu_available()
                && env != "0";

            s_isUseOfGpuForced = string.Equals(env, "force", StringComparison.OrdinalIgnoreCase);
        }
        //---------------------------------------------------------------------
        public static bool IsAvailable      => s_isAvailable;
        public static bool IsUseOfGpuForced => s_isUseOfGpuForced;
        //---------------------------------------------------------------------
        public static unsafe void CalculateSampleStats(Sample sample)
        {
            if (!s_isAvailable)
                throw new InvalidOperationException(Strings.Gpu_not_available);

            fixed (double* ptr = sample.Values)
            {
                SampleStats sampleStats = default;
                int errorCode = GpuMethods.gpu_sample_calc_stats(ptr, sample.Count, &sampleStats);

                if (errorCode != 0) ThrowGpuException(errorCode);

                sample.Mean         = sampleStats.Mean;
                sample.Max          = sampleStats.Max;
                sample.Min          = sampleStats.Min;
                sample.Delta        = sampleStats.Delta;
                sample.VarianceCore = sampleStats.VarianceCore;
                sample.Skewness     = sampleStats.Skewness;
                sample.Kurtosis     = sampleStats.Kurtosis;
            }
        }
        //---------------------------------------------------------------------
        private static void ThrowGpuException(int errorCode)
        {
            IntPtr ptr = GpuMethods.gpu_get_error_string(errorCode);
            string msg = $"CUDA runtime error: {Marshal.PtrToStringAnsi(ptr)}";
            throw new GpuException(msg);
        }
        //---------------------------------------------------------------------
        private static class GpuMethods
        {
            private const string LibName = "gfoidl-Stochastics-gpu";
            //---------------------------------------------------------------------
            [DllImport(LibName)]
            public static extern bool gpu_available();
            //---------------------------------------------------------------------
            [DllImport(LibName)]
            public static extern IntPtr gpu_get_error_string(int errorCode);
            //---------------------------------------------------------------------
            [DllImport(LibName)]
            public static extern unsafe int gpu_sample_calc_stats(double* sample, int sampleSize, SampleStats* sampleStats);
        }
        //---------------------------------------------------------------------
        [StructLayout(LayoutKind.Sequential)]
        private struct SampleStats
        {
            public double Mean;
            public double Max;
            public double Min;
            public double Delta;
            public double VarianceCore;
            public double Skewness;
            public double Kurtosis;
        }
    }
    //-------------------------------------------------------------------------
    /// <summary>
    /// Represents an error that is caused by the GPU.
    /// </summary>
    public class GpuException : Exception
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="GpuException" /> class.
        /// </summary>
        public GpuException() : base() { }

        /// <summary>
        /// Initializes a new instance of the <see cref="GpuException" /> class 
        /// with a specified error message.
        /// </summary>
        /// <param name="message">The message that describes the error.</param>
        public GpuException(string message) : base(message) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="GpuException" />
        /// class with a specified error message and a reference to the inner 
        /// exception that is the cause of this exception.
        /// </summary>
        /// <param name="message">The error message that explains the reason for the exception.</param>
        /// <param name="innerException">
        /// The exception that is the cause of the current exception, or a null reference (Nothing in Visual Basic) 
        /// if no inner exception is specified.
        /// </param>
        public GpuException(string message, Exception innerException) : base(message, innerException) { }
    }
}
