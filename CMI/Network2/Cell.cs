using static CMI.Utils;
using NumSharp;

namespace CMI.Network2
{
    public sealed class Cell : LSTM
    {
        public double x { get; set; }
        public NDArray ht_1 { get; set; }
        public NDArray ct_1 { get; set; }
        public NDArray ct { get; set; }
        public NDArray ht { get; set; }
        public double label { get; set; }
        public NDArray a { get; set; }
        public NDArray i { get; set; }
        public NDArray f { get; set; }
        public NDArray o { get; set; }

        public double dx { get; set; }
        public NDArray dht { get; set; }
        public NDArray dht_1 { get; set; }
        public NDArray dct { get; set; }
        public NDArray da { get; set; }
        public NDArray di { get; set; }
        public NDArray df { get; set; }
        public NDArray do_ { get; set; }

        public NDArray dloss { get; set; }

        public Cell(double x, NDArray ct_1, NDArray ht_1) : base()
        {
            this.x = x;
            this.ct_1 = ct_1;
            this.ht_1 = ht_1;
        }

        public void forward()
        {
            update_gate();
            forget_gate();
            output_gate();
        }

        public void backpropagation(double target, NDArray ht, NDArray next_dct, NDArray next_f)
        {
            dloss = this.ht - target;
            dht = dloss + ht;
            dct = dht * o * (1 - Tanh2(ct)) + next_dct * next_f;

            //da = dct * i * (1 - Math.Pow(a, 2));
            da = dct * i * (1 - np.power(a, 2));
            di = dct * a * i * (1 - i);
            df = dct * ct_1 * f * (1 - f);
            do_ = dht * Tanh(ct) * o * (1 - o);

            //dx = Wa * da + Wi * di + Wf * df + Wo * do_; // it's never used
            dht_1 = Ua * da + Ui * di + Uf * df + Uo * do_;
        }
                    
        private void update_gate()
        {
            a = Tanh(Wa * x + Ua * ht_1 + ba);
            i = Sigmoid(Wi * x + Ui * ht_1 + bi);
        }

        private void forget_gate()
        {
            f = Sigmoid(Wf * x + Uf * ht_1 + bf);
            ct = f * ct_1 + a * i;
        }

        private void output_gate()
        {
            o = Sigmoid(Wo * x + Uo * ht_1 + bo);
            ht = Tanh(ct) * o;
        }
    }
}
