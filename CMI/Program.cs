using CMI;
using CMI.GPU;
using CMI.Network;
using NumSharp;
using OpenCL;

static void print(object? text)
{
    Console.WriteLine(text.ToString());
}


LSTM lstm = new();
lstm.initialize();
double[] input = new double[] { 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39 };
double[] output = new double[] { 0.034, 0.035, 0.036, 0.037, 0.038, 0.039, 0.040 };

lstm.train(input, output, 100000000);





