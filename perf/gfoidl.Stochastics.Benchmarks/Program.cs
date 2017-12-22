using System;

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

            string arg = args[0];

            switch (arg)
            {
                case nameof(CalculateDeltaBenchmarks):
                    CalculateDeltaBenchmarks.Run();
                    break;
                default:
                    Console.WriteLine($"unknown benchmark '{arg}'");
                    break;
            }
        }
    }
}