using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using gfoidl.Stochastics.Builders;

namespace gfoidl.Stochastics.Statistics
{
    partial class Sample
    {
        public void AddRange(IEnumerable<double> values)
        {
            if (values == null) ThrowHelper.ThrowArgumentNull(nameof(values));

            if (values is double[] array)
            {
                _values = array;
                return;
            }
            else if (values is ICollection<double> collection)
            {
                int n = collection.Count;
                array = new double[n];
                collection.CopyTo(array, 0);
                _values = array;
                return;
            }

            double min       = double.MaxValue;
            double max       = double.MinValue;
            double avg       = 0;
            int count        = 0;
            var arrayBuilder = new ArrayBuilder<double>(true);

            foreach (double item in values)
                AddItem(item, ref arrayBuilder, ref count, ref min, ref max, ref avg);

            _values = arrayBuilder.ToArray();
            _min    = min;
            _max    = max;
            _mean   = avg / count;
        }
        //---------------------------------------------------------------------
        public IEnumerable<double> AddRangeWithIteration(IEnumerable<double> values)
        {
            if (values == null) ThrowHelper.ThrowArgumentNull(nameof(values));

            return Core();
            //-----------------------------------------------------------------
            IEnumerable<double> Core()
            {
                double min       = double.MaxValue;
                double max       = double.MinValue;
                double avg       = 0;
                int count        = 0;
                var arrayBuilder = new ArrayBuilder<double>(true);

                foreach (double item in values)
                {
                    AddItem(item, ref arrayBuilder, ref count, ref min, ref max, ref avg);

                    yield return item;
                }

                _values = arrayBuilder.ToArray();
                _min    = min;
                _max    = max;
                _mean   = avg / count;
            }
        }
        //---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddItem(
            double item,
            ref ArrayBuilder<double> arrayBuilder,
            ref int count,
            ref double min,
            ref double max,
            ref double avg)
        {
            arrayBuilder.Add(item);

            if (item < min) min = item;
            if (item > max) max = item;

            avg += item;
            count++;
        }
    }
}