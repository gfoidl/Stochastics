﻿using System;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.OutlierDetectionTests
{
    [TestFixture]
    public class Ctor : Base
    {
        [Test]
        public void Sample_is_null___throws_ArgumentNull()
        {
            Assert.Throws<ArgumentNullException>(() => new TestOutlierDetection(null));
        }
        //---------------------------------------------------------------------
        [Test]
        public void Sample_given___OK()
        {
            double[] values = { 1, 2, 3 };
            var sample      = new Sample(values);

            OutlierDetection sut = new TestOutlierDetection(sample);

            Assert.AreSame(sample, sut.Sample);
        }
    }
}
