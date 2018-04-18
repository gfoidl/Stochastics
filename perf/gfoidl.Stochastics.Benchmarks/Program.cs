using BenchmarkDotNet.Running;

namespace gfoidl.Stochastics.Benchmarks
{
    static class Program
    {
        static void Main(string[] args)
        {
            var switcher = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly);
            switcher.Run(args);
        }
    }
}