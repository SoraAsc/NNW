using System;
using Examples;

class Program
{
    static void Main(string[] args)
    {
        string choice = args.Length > 0 ? args[0].Trim().ToUpperInvariant() : string.Empty;

        if (string.IsNullOrEmpty(choice))
        {
            Console.WriteLine("Choose an example to execute:");
            Console.WriteLine("  AND  - Logic Gate AND");
            Console.WriteLine("  XOR  - Logic Gate XOR");
            Console.Write("Type AND or XOR (or run as an arg): ");
            choice = Console.ReadLine()?.Trim().ToUpperInvariant() ?? string.Empty;
        }

        try
        {
            switch (choice)
            {
                case "AND":
                    AndExample.Run();
                    break;
                case "XOR":
                    XorExample.Run();
                    break;
                default:
                    Console.WriteLine("Invalid Option. Use AND or XOR.");
                    break;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error while executing: {ex.Message}");
            Console.WriteLine(ex);
        }
    }
}
