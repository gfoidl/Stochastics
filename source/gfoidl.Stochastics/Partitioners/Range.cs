using System;

namespace gfoidl.Stochastics.Partitioners
{
    internal readonly struct Range : IEquatable<Range>
    {
        private static readonly Range _null = new Range(-1, -1);
        //---------------------------------------------------------------------
        public static ref readonly Range Null => ref _null;
        //---------------------------------------------------------------------
        public int Start { get; }
        public int End   { get; }
        public int Size => this.End - this.Start;
        //---------------------------------------------------------------------
        public Range(int start, int end)
        {
            this.Start = start;
            this.End   = end;
        }
        //---------------------------------------------------------------------
        public static implicit operator Range((int Start, int End) range)  => new Range(range.Start, range.End);
        public static implicit operator (int Start, int End) (Range range) => (range.Start, range.End);
        public static implicit operator Tuple<int, int>(Range range)       => Tuple.Create(range.Start, range.End);
        public static implicit operator Range(Tuple<int, int> range)       => new Range(range.Item1, range.Item2);
        //---------------------------------------------------------------------
        public override string ToString() => $"({this.Start}, {this.End}) Size: {this.Size}";
        //---------------------------------------------------------------------
        public bool Equals(Range other)
            => this.Start == other.Start
            && this.End   == other.End;
        //---------------------------------------------------------------------
        public override bool Equals(object obj) => obj is Range other && this.Equals(other);
        //---------------------------------------------------------------------
        public override int GetHashCode() => this.Start;    // is sufficient and fast
        //---------------------------------------------------------------------
        public static bool operator ==(Range r1, Range r2) => r1.Equals(r2);
        public static bool operator !=(Range r1, Range r2) => !r1.Equals(r2);
    }
}