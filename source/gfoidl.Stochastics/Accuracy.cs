namespace gfoidl.Stochastics
{
    internal static class Accuracy
    {
        private static double _epsilon = double.NaN;
        //---------------------------------------------------------------------
        internal static double Epsilon
        {
            get
            {
                if (double.IsNaN(_epsilon))
                {
                    double tau  = 1;
                    double walt = 1;
                    double wneu = 0;

                    while (wneu != walt)
                    {
                        tau *= 0.5;
                        wneu = walt + tau;
                    }

                    _epsilon = tau;
                }

                return _epsilon;
            }
        }
    }
}