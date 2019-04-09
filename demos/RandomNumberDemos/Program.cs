using System;
using System.Globalization;
using System.IO;
using System.Threading;
using gfoidl.Stochastics.RandomNumbers;

namespace RandomNumberDemos
{
    class Program
    {
        private const int N = 10_000;
        //---------------------------------------------------------------------
        static void Main()
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            var rnd                   = new Random();
            var randomNumberGenerator = new RandomNumberGenerator();

            using (StreamWriter sw = File.CreateText("results.txt"))
            {
                sw.WriteLine("System.Random;Uniform;Normal_1;Normal_05;Normal_2;Exponential_l1;Exponential_l05;Exponential_l2");

                for (int i = 0; i < N; ++i)
                {
                    double sysRnd      = rnd.NextDouble();
                    double uniform     = randomNumberGenerator.Uniform();
                    double normal1     = randomNumberGenerator.NormalDistributed(0, 1);
                    double normal2     = randomNumberGenerator.NormalDistributed(0, 0.5);
                    double normal3     = randomNumberGenerator.NormalDistributed(0, 2);
                    double exponential1 = randomNumberGenerator.ExponentialDistributed(1);
                    double exponential2 = randomNumberGenerator.ExponentialDistributed(0.5);
                    double exponential3 = randomNumberGenerator.ExponentialDistributed(2);

                    sw.WriteLine($"{sysRnd};{uniform};{normal1};{normal2};{normal3};{exponential1};{exponential2};{exponential3}");
                }
            }
        }
    }
}
