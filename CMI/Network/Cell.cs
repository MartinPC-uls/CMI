using static CMI.Utils;

namespace CMI.Network
{
    public sealed class Cell : LSTM
    {
        public double x { get; set; }
        public double ht_1 { get; set; }
        public double ct_1 { get; set; }
        public double ct { get; set; }
        public double ht { get; set; }
        public double a { get; set; }
        public double i { get; set; }
        public double f { get; set; }
        public double o { get; set; }

        public double dx { get; set; }
        public double dht { get; set; }
        public double dht_1 { get; set; }
        public double dct { get; set; }
        public double da { get; set; }
        public double di { get; set; }
        public double df { get; set; }
        public double do_ { get; set; }

        public double dloss { get; set; }

        public Cell(double x, double ct_1, double ht_1) : base()
        {
            this.x = x;
            this.ct_1 = ct_1;
            this.ht_1 = ht_1;
        }

        public void Forward()
        {
            UpdateGate();
            ForgetGate();
            OutputGate();
        }

        public void Backpropagation(double target, double ht, double next_dct, double next_f)
        {
            dloss = this.ht - target;
            dht = dloss + ht;
            dct = dht * o * (1 - Tanh2(ct)) + next_dct * next_f;

            da = dct * i * (1 - Math.Pow(a, 2));
            di = dct * a * i * (1 - i);
            df = dct * ct_1 * f * (1 - f);
            do_ = dht * Tanh(ct) * o * (1 - o);

            //dx = Wa * da + Wi * di + Wf * df + Wo * do_; // it's never used
            dht_1 = Ua * da + Ui * di + Uf * df + Uo * do_;
        }
                    
        private void UpdateGate()
        {
            a = Tanh(Wa * x + Ua * ht_1 + ba);
            i = Sigmoid(Wi * x + Ui * ht_1 + bi);
        }

        private void ForgetGate()
        {
            f = Sigmoid(Wf * x + Uf * ht_1 + bf);
            ct = f * ct_1 + a * i;
        }

        private void OutputGate()
        {
            o = Sigmoid(Wo * x + Uo * ht_1 + bo);
            ht = Tanh(ct) * o;
        }
    }
}
