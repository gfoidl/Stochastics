using System;
using System.Collections.Generic;

namespace gfoidl.Stochastics.Statistics
{
    public class SampleBuilder
    {
        public IEnumerable<double> Add(IEnumerable<double> values)
        {
            throw new NotImplementedException();
        }
        //---------------------------------------------------------------------
        public Sample GetSample()
        {
            throw new NotImplementedException();
        }
    }
    //-------------------------------------------------------------------------
    public static class SampleBuilderExtensions
    {
        public static IEnumerable<double> AddToSampleBuilder(this IEnumerable<double> values, SampleBuilder sampleBuilder)
            => sampleBuilder.Add(values);
    }
}