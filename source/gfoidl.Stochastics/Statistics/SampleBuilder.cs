﻿using System.Collections.Generic;
using System.Runtime.CompilerServices;
using gfoidl.Stochastics.Builders;
using gfoidl.Stochastics.Enumerators;

namespace gfoidl.Stochastics.Statistics
{
    public class SampleBuilder
    {
        // Must not be readonly, cf. https://gist.github.com/gfoidl/14b07dfe8ee5cb093f216f8a85759d88
        private ArrayBuilder<double> _arrayBuilder;
        private double               _min = double.MaxValue;
        private double               _max = double.MinValue;
        private double               _sum;
        private bool                 _canUseStats = true;
        //---------------------------------------------------------------------
        public SampleBuilder() => _arrayBuilder = new ArrayBuilder<double>(true);
        //---------------------------------------------------------------------
        public void Add(double item) => this.AddCore(item, ref _min, ref _max, ref _sum);
        //---------------------------------------------------------------------
        public void Add(IEnumerable<double> values)
        {
            if (values is double[] array)
            {
                _arrayBuilder.AddRange(array);
                _canUseStats = false;
                return;
            }

            double min = double.MaxValue;
            double max = double.MinValue;
            double sum = 0;

            foreach (double item in values)
                this.AddCore(item, ref min, ref max, ref sum);

            if (_canUseStats)
            {
                _min = min;
                _max = max;
                _sum = sum;
            }
        }
        //---------------------------------------------------------------------
        public IEnumerable<double> AddWithYield(IEnumerable<double> values)
        {
            if (values is double[] array)
                return this.AddWithYield(array);

            return Core();
            //-----------------------------------------------------------------
            IEnumerable<double> Core()
            {
                double min = double.MaxValue;
                double max = double.MinValue;
                double sum = 0;

                foreach (double item in values)
                {
                    this.AddCore(item, ref min, ref max, ref sum);
                    yield return item;
                }

                if (_canUseStats)
                {
                    _min = min;
                    _max = max;
                    _sum = sum;
                }
            }
        }
        //---------------------------------------------------------------------
        public ArrayEnumerable<double> AddWithYield(double[] array)
        {
            _arrayBuilder.AddRange(array);
            _canUseStats = false;

            return new ArrayEnumerable<double>(array);
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void AddCore(double item, ref double min, ref double max, ref double sum)
        {
            _arrayBuilder.Add(item);

            if (_canUseStats)
            {
                sum += item;
                if (item < min) min = item;
                if (item > max) max = item;
            }
        }
        //---------------------------------------------------------------------
        public Sample GetSample()
        {
            ref var arrayBuilder = ref _arrayBuilder;

            double[] values = arrayBuilder.ToArray();

            var sample = new Sample(values);

            if (_canUseStats)
            {
                sample.Min  = _min;
                sample.Max  = _max;
                sample.Mean = _sum / arrayBuilder.Count;
            }

            return sample;
        }
    }
    //-------------------------------------------------------------------------
    public static class SampleBuilderExtensions
    {
        public static IEnumerable<double> AddToSampleBuilder(this IEnumerable<double> values, SampleBuilder sampleBuilder)
            => sampleBuilder.AddWithYield(values);
    }
}