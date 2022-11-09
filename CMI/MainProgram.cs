using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
