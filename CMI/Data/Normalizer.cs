using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static CMI.Utils;

namespace CMI.Data
{
    public class Normalizer
    {
        private List<char[]> Dataset;
        public static List<int[]>? NormalizedDataset;
        public static List<char[]>? DenormalizedDataset;

        private static readonly double MIN = 32;
        private static readonly double MAX = 122;

        private static readonly double MODIFIER = 1000.0;

        public Normalizer(List<char[]> Dataset)
        {
            this.Dataset = Dataset;
            NormalizedDataset = new();
        }

        public static double[] Normalize(char[] x)
        {
            double limIzquierdo = 32;
            double limDerecho, centro = 122;
            List<double> valoresNormalizados = new List<double>();
            foreach (char c in x)
            {
                //print((c - limIzquierdo) / (centro - limIzquierdo));
                valoresNormalizados.Add((c - limIzquierdo) / (centro - limIzquierdo));
            }
            return valoresNormalizados.ToArray();
        }

        public static char Denormalize(double output)
        {
            double valor = output * (122 - 32) + 32;
            return (char)Math.Round(valor, 0);
        }
    }
}
