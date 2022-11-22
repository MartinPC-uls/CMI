using CMI;
using CMI.Network;
using NumSharp;

static void print(object? text)
{
    Console.WriteLine(text.ToString());
}

/*np.random.seed(1);

var x = np.array(new double[,,]
    {   
    {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 }
        },
    {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 }
        },
    {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 }
        }
    });

var a0 = np.random.randn(5, 10); // why is a0 5x10? should be 4x10 (4 notes, 10 examples)
var Wf = np.random.randn(5, 8);
var bf = np.random.randn(5, 1);
var Wi = np.random.randn(5, 8);
var bi = np.random.randn(5, 1);
var Wo = np.random.randn(5, 8);
var bo = np.random.randn(5, 1);
var Wc = np.random.randn(5, 8);
var bc = np.random.randn(5, 1);

var Wy = np.random.randn(2, 5);
var by = np.random.randn(2, 1);

List<NDArray> parameters = new List<NDArray>();
parameters.Add(Wf);
parameters.Add(bf);
parameters.Add(Wi);
parameters.Add(bi);
parameters.Add(Wc);
parameters.Add(bc);
parameters.Add(Wo);
parameters.Add(bo);
parameters.Add(Wy);
parameters.Add(by);

RNN rnn = new(parameters, 1);

var values = rnn.lstm_forward(x, a0);

// print predicted value
//print("predicted: " + values.Item2.Shape);
//Console.WriteLine(values.Item2.ToString());
var predicted_output = values.Item2;
var actual_output = np.array(new double[,,]
    {
    {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 }
        },
    {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 }
        },
    {
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 },
        { 1, 2, 3, 4 }
        }
    });
var da = values.Item1;

var gradients = rnn.lstm_backward(da, values.Item4.Item1);

print(x.Shape);
print(actual_output.Shape);
var d_a0 = np.zeros(5, 10);
var __x__ = x;

// repeat everything 1000 times
for (int i = 1; i <= 100000; i++)
{
    // forward pass
    //print(rnn.Wf[0][0].ToString());
    var _values = rnn.lstm_forward(x, d_a0);
    // print predicted value
    //print("predicted: " + values.Item2.Shape);
    //Console.WriteLine(values.Item2.ToString());
    var _predicted_output = _values.Item1;
    var _predicted_output_softmax = _values.Item2;
    //print(_predicted_output[0][0][0].ToString());

    // calculate loss
    var _loss = np.square(_predicted_output_softmax - actual_output);

    // print loss
    if (i % 100 == 0)
        print(_predicted_output);
        //print("i: " + i + ", " + "loss: " + _loss);

    // backpropagation
    var _da = _values.Item1;

    var _gradients = rnn.lstm_backward(_da, _values.Item4.Item1);
    d_a0 = _gradients[1];

    // update parameters
    rnn.update_parameters_lstm(_gradients, 0.01);
}*/

LSTM lstm = new();
lstm.initialize();
//lstm.initialize();
// sequential input
double[] input = new double[] { 1, 2, 3 };
// sequential output
double[] output = new double[] { 2, 3, 4 };
lstm.train(input, output, 10000000);