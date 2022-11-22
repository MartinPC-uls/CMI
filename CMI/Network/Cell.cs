using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CMI.Network
{
    public sealed class Cell : LSTM
    {
        public double input { get; set; }
        public double prev_long { get; set; }
        public double prev_short { get; set; }

        public double next_long { get; set;  }
        public double next_short { get; set; }

        public double fg_value { get; set; }
        public double ug_value { get; set; }
        // from ug_value: pltm and pmr
        public double pltm_value { get; set; }
        public double pmr_value { get; set; }

        public double og_value { get; set; }
        // output prediction
        public double output { get; set; }

        public Cell(double input, double prev_long, double prev_short) : base()
        {
            this.input = input;
            this.prev_long = prev_long;
            this.prev_short = prev_short;

            next_long = prev_long;
            next_short = prev_short;

            forward();
        }

        private void forward()
        {
            forget_gate();
            update_gate();
            output_gate();
        }

        public List<double> backpropagation(double original_value, double predicted_value)
        {
            /* 
             Returns:
              Forget Gate Gradients:
                dWif w.r.t Wif
                dWsf w.r.t Wsf
                dbf  w.r.t bf
            
              Update Gate Gradients:
                dWipltm w.r.t Wipltm
                dWspltm w.r.t Wspltm
                dbpltm  w.r.t bpltm
                dWipmr  w.r.t Wipmr
                dWspmr  w.r.t Wspmr
                dbpmr   w.r.t bpmr

              Output Gate Gradients:
                dWso w.r.t Wso
                dWio w.r.t Wio
                dbo  w.r.t bo
            
             * w.r.t: With Respect To
             */

            double E_delta = 0.5 * Math.Pow(original_value - predicted_value, 2);

            List<double> fg_gradients = getForgetGateGradients(E_delta);
            List<double> ug_gradients = getUpdateGateGradients(E_delta);
            List<double> og_gradients = getOutputGateGradients(E_delta);

            List<double> gradients = new();
            gradients.AddRange(fg_gradients);
            gradients.AddRange(ug_gradients);
            gradients.AddRange(og_gradients);

            return gradients;
        }

        private List<double> getForgetGateGradients(double E_delta)
        {
            List<double> gradients = new();

            var f_value = Wsf * prev_short + Wif * input + bf;
            var derivative_fg = E_delta * og_value * (1 - Math.Pow(tanh(next_long), 2)) * prev_long *
                sigmoid(f_value) * (1 - sigmoid(f_value));
            
            var dWif = derivative_fg * input;
            var dWsf = derivative_fg * prev_short;
            var dbf = derivative_fg;

            gradients.Add(dWif);
            gradients.Add(dWsf);
            gradients.Add(dbf);

            return gradients;
        }

        private List<double> getUpdateGateGradients(double E_delta)
        {
            List<double> gradients = new();

            var gradients_pmr = getPotentialMemoryToRememberGradients(E_delta);
            var gradients_pltm = getPotentialLongTermMemoryGradients(E_delta);

            gradients.AddRange(gradients_pmr);
            gradients.AddRange(gradients_pltm);

            return gradients;
        }

        private List<double> getPotentialMemoryToRememberGradients(double E_delta)
        {
            List<double> gradients = new();
            var pmr_value = Wipmr * input + Wspmr * prev_short + bpmr;
            var derivative_pmr = E_delta * og_value * (1 - Math.Pow(tanh(next_long), 2)) * this.pmr_value *
                sigmoid(pmr_value) * (1 - sigmoid(pmr_value));
            
            var dWimpr = derivative_pmr * input;
            var dWspmr = derivative_pmr * prev_short;
            var dbpmr = derivative_pmr;

            gradients.Add(dWimpr);
            gradients.Add(dWspmr);
            gradients.Add(dbpmr);

            return gradients;
        }
        private List<double> getPotentialLongTermMemoryGradients(double E_delta)
        {
            List<double> gradients = new();

            var pltm_value = Wipltm * input + Wspltm * prev_short + bpltm;
            var derivative_pltm = E_delta * og_value * (1 - Math.Pow(tanh(next_long), 2)) * this.pltm_value *
                (1 - Math.Pow(tanh(pltm_value), 2));
            
            var dWipltm = derivative_pltm * input;
            var dWspltm = derivative_pltm * prev_short;
            var dbpltm = derivative_pltm;

            gradients.Add(dWipltm);
            gradients.Add(dWspltm);
            gradients.Add(dbpltm);

            return gradients;
        }

        private List<double> getOutputGateGradients(double E_delta)
        {
            List<double> gradients = new();

            var o_value = Wio * input + Wso * prev_short + bo;
            var derivative_og = E_delta * tanh(next_long) * sigmoid(o_value) * (1 - sigmoid(o_value));
            
            var dWio = derivative_og * input;
            var dWso = derivative_og * prev_short;
            var dbo = derivative_og;

            gradients.Add(dWio);
            gradients.Add(dWso);
            gradients.Add(dbo);

            return gradients;
        }

        private void forget_gate()
        {
            var result = sigmoid(Wsf * prev_short + Wif * input + bf);

            fg_value = result;

            next_long *= result;
        }

        private double potential_long_term_memory()
        {
            var result = tanh(Wipltm * input + Wspltm * prev_short + bpltm);

            pltm_value = result;

            return result;
        }
        private double potential_memory_to_remember()
        {
            var result = sigmoid(Wipmr * input + Wspmr * prev_short + bpmr);

            pmr_value = result;

            return result;
        }
        private void update_gate()
        {
            var pltm = potential_long_term_memory();
            var pmr = potential_memory_to_remember();

            var result = pltm * pmr;

            ug_value = result;

            next_long += result;
        }
        private void output_gate()
        {
            var psmr = sigmoid(Wio * input + Wso * prev_short + bo);
            var pstm = tanh(prev_long);
            
            var result = psmr * pstm;

            next_short = result;

            og_value = next_short;
            
            output = next_short;
        }
    }
}
