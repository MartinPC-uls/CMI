using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CMI.Network;
using static CMI.Utils;
using static CMI.Data.Normalizer;
using MusicXML_Parser;
using System.Xml;

namespace CMI
{
    public class MainProgram
    {
        public string Name { get; set; }
        public string Key { get; set; }
        public int Tempo { get; set; }
        public string Autor { get; set; }
        public string NotaPrincipal { get; set; }
        public void print(string text)
        {
            Console.WriteLine(text);
        }
        public MainProgram()
        {
            print("BIENVENIDO A LA IA GENERADORA DE MÚSICA");
            Console.Write("¿Quiere elegir usted los parámetros? (si/no): ");
            string opcion = Console.ReadLine();
            if (opcion.Equals("si"))
            {
                Console.Write("Escriba un nombre para la composición musical: ");
                string? name = Console.ReadLine();
                Name = name;
                Console.Clear();
                print("###################################################################");
                print("\t\t\t\t" + name);
                print("###################################################################\n");
                print("Elija una de las siguientes tonalidades musicales");
                print("1. C Major\t\t13. A minor");
                print("2. C# Major\t\t14. A# minor");
                print("3. D Major\t\t15. B minor");
                print("4. D# Major\t\t16. C minor");
                print("5. E Major\t\t17. C# minor");
                print("6. F Major\t\t18. D minor");
                print("7. F# Major\t\t19. D# minor");
                print("8. G Major\t\t20. E minor");
                print("9. G# Major\t\t21. F minor");
                print("10. A Major\t\t22. F# minor");
                print("11. A# Major\t\t23. G minor");
                print("12. B Major\t\t24. G# minor\n");
                Console.Write("Escriba el número o el nombre de la tonalidad musical: ");
                string? tonalidad = Console.ReadLine();
                string _tonalidad = Tonalidad(tonalidad);
                Key = _tonalidad;
                Console.Write("Elija un tempo: ");
                string? tempo = Console.ReadLine();
                int _tempo = int.Parse(tempo);
                Tempo = _tempo;
                Console.Write("Ingrese nombre del autor: ");
                string? autor = Console.ReadLine();
                Autor = autor;
                Console.Clear();
                print("###################################################################");
                print("\t\t\tNombre de la pieza musical: " + Name);
                print("\t\t\tTonalidad: " + Key);
                print("\t\t\tTempo: " + Tempo + " bpm");
                print("\t\t\tTiempo: 4/4");
                print("\t\t\tAutor: " + Autor);
                print("###################################################################");
                print("Elija una de las siguientes notas musicales con la que desea que empiece la pieza musical:\n");
                print(" C#  D#    F#  G#  A#");
                print("C  D    E F   G  A  B\n\n");
                Console.Write(">>: ");
                string? nota = Console.ReadLine();

                InicializarModelo("".ToCharArray());
            }
            else
            {
                Name = "CMI-generated";
                Random random = new();
                Key = Tonalidad(random.Next(1, 25).ToString());
                Tempo = random.Next(100, 151);
                Autor = "IA CMI";
                InicializarModelo("".ToCharArray());
            }
            

        }

        public void InicializarModelo(char[] inputs)
        {
            inputs = " `^][` `^][[ YWWYV V JV JRVJVJRVVbV^b3:C3:C3:TRTRTR^7>F7>F7>QT]Q]QT]R^RV^5<E5<E5<TRTRTR^7>F7>F7>TW`R^Q]OW[T[`'3:C3:C3:T[`R^".Reverse().ToArray();
            // make inputs a random generation of characters
            Random random = new();
            for (int i = 0; i < 100; i++)
            {
                inputs[i] = (char)random.Next(32, 121);
            }
            double[] _inputs = Normalize(inputs);
            LSTM lstm = new();

            lstm.LoadParameters(AppDomain.CurrentDomain.BaseDirectory + "scale1.txt");

            lstm.Prediction(_inputs);

            SheetConfiguration music = new SheetConfiguration()
            {
                WorkTitle = Name,
                Composer = Autor,
                Accidental = true,
                Beam = true,
                Print_NewPage = true,
                Print_NewSystem = true,
                Stem = true,
                WordFont = "FreeSerif",
                WordFontSize = 10,
                LyricFont = "FreeSerif",
                LyricFontSize = 11,
                CreditWords_Title = "Partitura generada por CMI",
                CreditWords_Subtitle = "",
                CreditWords_Composer = "Autores CMI: Felipe Barrientos, Martín Pizarro, Roberto Verdugo.",
                StaffLayout = 2,
                StaffDistance = 65,
                Divisions = 24,
                Tempo = Tempo,
                TimeBeats = 4,
                TimeBeatType = 4
            };
            XmlDocument score = music.Init(Name + ".xml", MusicXML_Parser.Music.Key.CMajor);
            //XmlDocument score = sheet.Load(AppDomain.CurrentDomain.BaseDirectory + "score.xml");
            XmlNode node = score.SelectSingleNode("score-partwise/part[@id='P1']");
            XmlNode measure = music.AddMeasure(node);

            foreach (var predicted in lstm.Predicted_Output)
            {
                music.Add(predicted, 1, measure);
            }
        }

        public string Tonalidad(string? tonalidad)
        {
            switch (tonalidad)
            {
                case "1":
                case "C Major":
                    return "C Major";
                case "2":
                case "C# Major":
                    return "C# Major";
                case "3":
                case "D Major":
                    return "D Major";
                case "4":
                case "D# Major":
                    return "D# Major";
                case "5":
                case "E Major":
                    return "E Major";
                case "6":
                case "F Major":
                    return "F Major";
                case "7":
                case "F# Major":
                    return "F# Major";
                case "8":
                case "G Major":
                    return "G Major";
                case "9":
                case "G# Major":
                    return "G# Major";
                case "10":
                case "A Major":
                    return "A Major";
                case "11":
                case "A# Major":
                    return "A# Major";
                case "12":
                case "B Major":
                    return "B Major";
                case "13":
                case "A minor":
                    return "A minor";
                case "14":
                case "A# minor":
                    return "A# minor";
                case "15":
                case "B minor":
                    return "B minor";
                case "16":
                case "C minor":
                    return "C minor";
                case "17":
                case "C# minor":
                    return "C# minor";
                case "18":
                case "D minor":
                    return "D minor";
                case "19":
                case "D# minor":
                    return "D# minor";
                case "20":
                case "E minor":
                    return "E minor";
                case "21":
                case "F minor":
                    return "F minor";
                case "22":
                case "F# minor":
                    return "F# minor";
                case "23":
                case "G minor":
                    return "G minor";
                case "24":
                case "G# minor":
                    return "G# minor";
                default:
                    return "INVALID";
            }
        }
    }
}
