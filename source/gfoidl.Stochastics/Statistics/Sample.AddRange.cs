using System.Collections.Generic;
using System.Diagnostics;
using gfoidl.Stochastics.Builders;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        public IEnumerable<double> AddRange(IEnumerable<double> values)
        {
            if (values == null) ThrowHelper.ThrowArgumentNull(nameof(values));

            return Core();
            //-----------------------------------------------------------------
            IEnumerable<double> Core()
            {
                System.Console.WriteLine(values.GetType());

                double min       = double.MaxValue;
                double max       = double.MinValue;
                double avg       = 0;
                int count        = 0;
                var arrayBuilder = new ArrayBuilder<double>(true);

                foreach (double item in values)
                {
                    arrayBuilder.Add(item);

                    if (item < min) min = item;
                    if (item > max) max = item;

                    avg += item;
                    count++;

                    yield return item;
                }

                _values = arrayBuilder.ToArray();
                _min    = min;
                _max    = max;
                _mean   = avg / count;
            }
        }
    }
}