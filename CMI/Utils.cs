using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CMI
{
    public class Utils
    {
        public static void print(object? obj)
        {
            Console.WriteLine(obj.ToString());
        }
        public static double softmax(double[] x)
        {
            double e_x = 0;
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                e_x += Math.Exp(x[i]);
            }
            for (int i = 0; i < x.Length; i++)
            {
                sum += Math.Exp(x[i]) / e_x;
            }
            return sum;
        }
        public static double softmax(double x)
        {
            double e_x = 0;
            double sum = 0;
            e_x += Math.Exp(x);
            sum += Math.Exp(x) / e_x;
            print("sum: " + e_x);
            return sum;
        }
        public static double sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        public static double tanh(double x)
        {
            var result = Math.Tanh(x);

            return result;
        }
        public static double tanh2(double x)
        {
            var result = Math.Pow(Math.Tanh(x), 2);

            return result;
        }
        public static double generateRandom()
        {
            Random random = new Random();
            return random.NextDouble() - random.NextDouble();
        }
    }
}
