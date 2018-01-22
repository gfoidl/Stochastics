﻿using System.Collections.Generic;
using gfoidl.Stochastics.Builders;

namespace gfoidl.Stochastics.Statistics
{
    public class SampleBuilder
    {
        // Must not be readonly, cf. https://gist.github.com/gfoidl/14b07dfe8ee5cb093f216f8a85759d88
        private ArrayBuilder<double> _arrayBuilder;
        private double _min = double.MaxValue;
        private double _max = double.MinValue;
        private double _sum;
        //---------------------------------------------------------------------
        public SampleBuilder() => _arrayBuilder = new ArrayBuilder<double>(true);
        //---------------------------------------------------------------------
        public void Add(double item)
        {
            _arrayBuilder.Add(item);

            _sum += item;
            if (item < _min) _min = item;
            if (item > _max) _max = item;
        }
        //---------------------------------------------------------------------
        public void Add(IEnumerable<double> values)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            double sum = 0;

            foreach (double item in values)
            {
                _arrayBuilder.Add(item);

                sum += item;
                if (item < min) min = item;
                if (item > max) max = item;
            }

            _min = min;
            _max = max;
            _sum = sum;
        }
        //---------------------------------------------------------------------
        public IEnumerable<double> AddWithYield(IEnumerable<double> values)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            double sum = 0;

            foreach (double item in values)
            {
                _arrayBuilder.Add(item);

                sum += item;
                if (item < min) min = item;
                if (item > max) max = item;

                yield return item;
            }

            _min = min;
            _max = max;
            _sum = sum;
        }
        //---------------------------------------------------------------------
        public Sample GetSample()
        {
            ref var arrayBuilder = ref _arrayBuilder;

            double[] values = arrayBuilder.ToArray();

            return new Sample(values)
            {
                Min  = _min,
                Max  = _max,
                Mean = _sum / arrayBuilder.Count
            };
        }
    }
    //-------------------------------------------------------------------------
    public static class SampleBuilderExtensions
    {
        public static IEnumerable<double> AddToSampleBuilder(this IEnumerable<double> values, SampleBuilder sampleBuilder)
            => sampleBuilder.AddWithYield(values);
    }
}