using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Reflection;

#if NETCOREAPP2_1
using System.Runtime.Intrinsics.X86;
#endif

namespace gfoidl.Stochastics.Benchmarks
{
    static class Program
    {
        static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("one and only arg has to be name of benchmark");
                PrintAvailableBenchmarks();
                Environment.Exit(1);
            }

            PrintSimdInfo();

            string type          = args[0];
            IBenchmark benchmark = GetBenchmark(type);

            if (benchmark == null)
            {
                Console.WriteLine($"unknown benchmark '{type}'");
                PrintAvailableBenchmarks();
                Environment.Exit(2);
            }

            benchmark.Run();

            if (Debugger.IsAttached)
            {
                Console.WriteLine("\nEnd.");
                Console.ReadKey();
            }
        }
        //---------------------------------------------------------------------
        private static void PrintSimdInfo()
        {
            Console.WriteLine($"{nameof(Vector.IsHardwareAccelerated)}: {Vector.IsHardwareAccelerated}");
            Console.WriteLine($"SIMD-size: {Vector<double>.Count * sizeof(double) * 8}");
            Console.WriteLine($"Vector<double>.Count: {Vector<double>.Count}");
#if NETCOREAPP2_1
            Console.WriteLine($"Sse : {Sse.IsSupported}");
            Console.WriteLine($"Sse2: {Sse2.IsSupported}");
            Console.WriteLine($"Avx : {Avx.IsSupported}");
            Console.WriteLine($"Avx2: {Avx2.IsSupported}");
#endif
            Console.WriteLine();
        }
        //---------------------------------------------------------------------
        private static IBenchmark GetBenchmark(string type)
        {
            Assembly assembly = typeof(Program).Assembly;

            Type benchmarkType = assembly
                .DefinedTypes
                .FirstOrDefault(t => typeof(IBenchmark).IsAssignableFrom(t) && t.Name == type);

            if (benchmarkType == null) return null;

            return Activator.CreateInstance(benchmarkType) as IBenchmark;
        }
        //---------------------------------------------------------------------
        private static void PrintAvailableBenchmarks()
        {
            Assembly assembly = typeof(Program).Assembly;

            var types = assembly
                .DefinedTypes
                .Where(t => typeof(IBenchmark).IsAssignableFrom(t) && t != typeof(IBenchmark))
                .OrderBy(t => t.Name);

            Console.WriteLine("\nAvailable benchmarks:");
            foreach (Type type in types)
                Console.WriteLine($"\t{type.Name}");
        }
    }
}
