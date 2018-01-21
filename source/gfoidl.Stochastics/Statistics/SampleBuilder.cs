using System.Collections.Generic;
using gfoidl.Stochastics.Builders;

namespace gfoidl.Stochastics.Statistics
{
    public class SampleBuilder
    {
        private double[] _values;
        private double _min;
        private double _max;
        private double _avg;
        //---------------------------------------------------------------------
        public IEnumerable<double> Add(IEnumerable<double> values)
        {
            var arrayBuilder = new ArrayBuilder<double>(true);
            int count        = 0;
            double min       = double.MaxValue;
            double max       = double.MinValue;
            double avg       = 0;

            foreach (double item in values)
            {
                arrayBuilder.Add(item);

                count++;
                avg += item;
                if (item < min) min = item;
                if (item > max) max = item;

                yield return item;
            }

            _values = arrayBuilder.ToArray();
            _min    = min;
            _max    = max;
            _avg    = avg / count;
        }
        //---------------------------------------------------------------------
        public Sample GetSample()
        {
            return new Sample(_values)
            {
                Min  = _min,
                Max  = _max,
                Mean = _avg
            };
        }
    }
    //-------------------------------------------------------------------------
    public static class SampleBuilderExtensions
    {
        public static IEnumerable<double> AddToSampleBuilder(this IEnumerable<double> values, SampleBuilder sampleBuilder)
            => sampleBuilder.Add(values);
    }
}