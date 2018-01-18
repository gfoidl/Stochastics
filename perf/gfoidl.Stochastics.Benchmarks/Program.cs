using System;
using System.Diagnostics;
using System.Numerics;

namespace gfoidl.Stochastics.Benchmarks
{
    static class Program
    {
        static void Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.WriteLine("one and only arg has to be name of benchmark");
                Environment.Exit(1);
            }

            Console.WriteLine($"{nameof(Vector.IsHardwareAccelerated)}: {Vector.IsHardwareAccelerated}");

            string arg = args[0];

            switch (arg)
            {
                case nameof(CalculateDeltaBenchmarks):
                    CalculateDeltaBenchmarks.Run();
                    break;
                case nameof(CalculateVarianceCoreBenchmarks):
                    CalculateVarianceCoreBenchmarks.Run();
                    break;
                case nameof(CalculateKurtosisBenchmarks):
                    CalculateKurtosisBenchmarks.Run();
                    break;
                case nameof(AutoCorrelationBenchmarks):
                    AutoCorrelationBenchmarks.Run();
                    break;
                case nameof(AutoCorrelationPartitionerBenchmarks):
                    AutoCorrelationPartitionerBenchmarks.Run();
                    break;
                case nameof(LoopSimdBenchmarks):
                    LoopSimdBenchmarks.Run();
                    break;
                case nameof(CalculateAverageBenchmarks):
                    CalculateAverageBenchmarks.Run();
                    break;
                case nameof(CombinedStatsGroup1Benchmarks):
                    CombinedStatsGroup1Benchmarks.Run();
                    break;
                case nameof(CombinedStatsGroup2Benchmarks):
                    CombinedStatsGroup2Benchmarks.Run();
                    break;
                case nameof(CombinedStatsGroup3Benchmarks):
                    CombinedStatsGroup3Benchmarks.Run();
                    break;
                case nameof(EnumerableWrapperBenchmarks):
                    EnumerableWrapperBenchmarks.Run();
                    break;
                case nameof(ListWrapperBenchmarks):
                    ListWrapperBenchmarks.Run();
                    break;
                default:
                    Console.WriteLine($"unknown benchmark '{arg}'");
                    break;
            }

            if (Debugger.IsAttached)
            {
                Console.WriteLine("\nEnd.");
                Console.ReadKey();
            }
        }
    }
}