using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;

namespace CMI
{
    public class RNN2 : Utils
    {
        private double input;
        private double prev_long { get; set; }
        private double prev_short { get; set; }

        private double Wsf; // Weight of short-term memory to forget gate
        private double Wif; // Weight of input to forget gate
        private double bf; // Bias of forget gate

        private double Wipltm; // Weight of input to potential long-term memory
        private double Wspltm; // Weight of short-term memory to potential long-term memory
        private double bpltm; // Bias of potential long-term memory

        private double Wipmr; // Weight of input to potential memory to remember
        private double Wspmr; // Weight of short-term memory to potential memory to remember
        private double bpmr; // Bias of potential memory to remember

        private double Wso; // Weight of short-term memory to output gate
        private double Wio; // Weight of input to output gate
        private double bo; // Bias of output gate

        private double[] sequential_data;

        private List<double> time_steps_outputs;

        public RNN2(double input, double prev_long, double prev_short)
        {
            this.input = input;
            this.prev_long = prev_long;
            this.prev_short = prev_short;
        }

        public void initialize()
        {
            Wsf = np.random.randn(1).ToArray<double>()[0];
            Wif = np.random.randn(1).ToArray<double>()[0];
            bf = np.random.randn(1).ToArray<double>()[0];

            Wipltm = np.random.randn(1).ToArray<double>()[0];
            Wspltm = np.random.randn(1).ToArray<double>()[0];
            bpltm = np.random.randn(1).ToArray<double>()[0];

            Wipmr = np.random.randn(1).ToArray<double>()[0];
            Wspmr = np.random.randn(1).ToArray<double>()[0];
            bpmr = np.random.randn(1).ToArray<double>()[0];

            Wso = np.random.randn(1).ToArray<double>()[0];
            Wio = np.random.randn(1).ToArray<double>()[0];
            bo = np.random.randn(1).ToArray<double>()[0];

            time_steps_outputs = new List<double>();
        }

        private double forget_gate()
        {
            var result = sigmoid(Wsf * prev_short + Wif * input + bf);

            return result;
        }
        private double potential_long_term_memory()
        {
            var result = tanh(Wipltm * input + Wspltm * prev_short + bpltm);

            return result;
        }
        private double potential_memory_to_remember()
        {
            var result = sigmoid(Wipmr * input + Wspmr * prev_short + bpmr);

            return result;
        }
        private double update_gate()
        {
            var pltm = potential_long_term_memory();
            var pmr = potential_memory_to_remember();

            var result = pltm * pmr;

            return result;
        }
        private double output_gate()
        {
            var psmr = sigmoid(Wio * input + Wso * prev_short + bo);
            var pstm = tanh(prev_long);

            var result = psmr * pstm;

            return result;
        }

        public double prediction(double[] sequential_data)
        {
            this.sequential_data = sequential_data;
            int time_steps = sequential_data.Length;

            // initial long and short term memories
            double ltm = 0;
            double stm = 0;
            for (int i = 0; i < time_steps; i++)
            {
                var values = lstm_forward(sequential_data[i], ltm, stm);
                ltm = values[0];
                stm = values[1];
            }

            return stm;
        }
        public List<double> lstm_forward(double input, double prev_long, double prev_short)
        {
            this.input = input;
            this.prev_long = prev_long;
            this.prev_short = prev_short;

            var fg = forget_gate();
            this.prev_long *= fg;
            var ug = update_gate(); // or input gate
            this.prev_long += ug;
            var og = output_gate();

            var next_long = this.prev_long;
            var next_short = og;

            List<double> values = new List<double>();
            values.Add(next_long);
            values.Add(next_short);

            return values;
        }
        public List<double> lstm_backward(double dnext_short, double dnext_long)
        {
            var og = output_gate();
            var dprev_short = dnext_short * og;
            var _dprev_long = dnext_long;

            var pltm = potential_long_term_memory();
            var pmr = potential_memory_to_remember();
            var dpltm = _dprev_long * pmr;
            var dpmr = _dprev_long * pltm;
            var dprev_long = _dprev_long * (1 - pmr * pmr);

            var dWipltm = dpltm * (1 - pltm * pltm) * input;
            var dWspltm = dpltm * (1 - pltm * pltm) * prev_short;
            var dbpltm = dpltm * (1 - pltm * pltm);

            var dWipmr = dpmr * pmr * (1 - pmr) * input;
            var dWspmr = dpmr * pmr * (1 - pmr) * prev_short;
            var dbpmr = dpmr * pmr * (1 - pmr);

            var dWsf = dprev_short * prev_short * (1 - prev_short * prev_short) * Wsf;
            var dWif = dprev_short * prev_short * (1 - prev_short * prev_short) * Wif;
            var dbf = dprev_short * prev_short * (1 - prev_short * prev_short);

            var dWso = dprev_short * prev_short * (1 - prev_short * prev_short) * Wso;
            var dWio = dprev_short * prev_short * (1 - prev_short * prev_short) * Wio;
            var dbo = dprev_short * prev_short * (1 - prev_short * prev_short);

            List<double> values = new List<double>();
            values.Add(dWsf);
            values.Add(dWif);
            values.Add(dbf);
            values.Add(dWipltm);
            values.Add(dWspltm);
            values.Add(dbpltm);
            values.Add(dWipmr);
            values.Add(dWspmr);
            values.Add(dbpmr);
            values.Add(dWso);
            values.Add(dWio);
            values.Add(dbo);

            return values;
        }
    }
}
