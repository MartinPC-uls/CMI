using NumSharp;

namespace CMI
{
    public class Utils
    {
        public static void Print(object? obj, bool skipLine = true)
        {
            if (obj is double[,] matrix)
            {
                Console.Write("[");
                for (int i = 0; i < matrix.GetLength(0); i++)
                {
                    Console.Write("[");
                    for (int j = 0; j < matrix.GetLength(1); j++)
                    {
                        Console.Write(matrix[i, j]);
                        if (j != matrix.GetLength(1) - 1)
                            Console.Write("\n");
                    }
                    Console.Write("]");
                    if (i != matrix.GetLength(0) - 1)
                        Console.Write("\n ");
                }
                Console.Write("]");
                return;
            }
            else if (obj is double[] vector)
            {
                Console.Write("[");
                for (int i = 0; i < vector.Length; i++)
                {
                    Console.Write(vector[i]);
                    if (i != vector.Length - 1)
                        Console.Write(" ");
                }
                Console.Write("]");
                Console.WriteLine();
                return;
            }

            if (!skipLine)
            {
                Console.Write(obj.ToString());
                return;
            }
            Console.WriteLine(obj.ToString());
        }
        public static double Softmax(double[] x)
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
        public static double Softmax(double x)
        {
            double e_x = 0;
            double sum = 0;
            e_x += Math.Exp(x);
            sum += Math.Exp(x) / e_x;
            Print("sum: " + e_x);
            return sum;
        }
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
        public static NDArray Sigmoid(NDArray x)
        {
            int shape1 = x.Shape[0];
            int shape2 = x.Shape[1];
            NDArray result = np.zeros((shape1, shape2));
            for (int i = 0; i < shape1; i++)
            {
                for (int j = 0; j < shape2; j++)
                {
                    double value = x[i, j];
                    result[i, j] = 1 / (1 + Math.Exp(-value));
                }
            }
            return result;
        }
        public static double Tanh(double x)
        {
            var result = Math.Tanh(x);

            return result;
        }

        public static NDArray Tanh(NDArray x)
        {
            return np.tanh(x);
        }
        public static double Tanh2(double x)
        {
            var result = Math.Pow(Math.Tanh(x), 2);

            return result;
        }
        public static NDArray Tanh2(NDArray x)
        {
            return np.tanh(x) * np.tanh(x);
        }
        public static double GenerateRandom()
        {
            Random random = new();
            Print("r: " + (random.NextDouble() - random.NextDouble()));
            return random.NextDouble() - random.NextDouble();
        }

        public static double GenerateXavierRandom()
        {
            // use Xavier initialization
            Random random = new();
            var value = (random.NextDouble() - random.NextDouble()) * Math.Sqrt(2.0 / 2.0);

            Print("GENERATED: " + value);
            return value;
        }

        public static double Round(double value)
        {
            var decPlaces = (int)(((decimal)value % 1) * 100);
            var integralValue = (int)value;

            if (decPlaces >= 75)
            {
                return integralValue + 1;
            }
            else
            {
                return integralValue;
            }
        }

        public static double[] GenerateRandomMatrix(int size)
        {
            Random random = new();
            double[] result = new double[size];
            for (int i = 0; i < size; i++)
            {
                result[i] = random.NextDouble() - random.NextDouble();
            }
            return result;
        }

        public static double[,] GenerateRandomMatrix(int xSize, int ySize)
        {
            Random random = new();
            double[,] result = new double[xSize, ySize];
            for (int i = 0; i < result.GetLength(0); i++)
            {
                for (int j = 0; j < result.GetLength(1); j++)
                {
                    result[i, j] = random.NextDouble() - random.NextDouble();
                }
            }
            return result;
        }
    }
}
