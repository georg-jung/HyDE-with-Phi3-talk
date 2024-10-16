// See https://aka.ms/new-console-template for more information
using System.Diagnostics;
using CorpusEmbedder;

var embedder = await Embedder.CreateBgeSmallEn1dot5Async();
using var cts = new CancellationTokenSource();
var lastReport = 0L;
var lastTokenizedSegments = 0;
var taskName = "";

void ReportProgress((int TokenizedSegments, int MdFilesCount, int RemainingFiles) x) 
{
    if (Stopwatch.GetElapsedTime(lastReport).TotalSeconds is double secs && secs < 5)
    {
        return;
    }

    lastReport = Stopwatch.GetTimestamp();
    var segmentsPerSec = (x.TokenizedSegments - lastTokenizedSegments) / secs;
    lastTokenizedSegments = x.TokenizedSegments;
    Console.WriteLine($"{taskName}: Tokenized {x.TokenizedSegments} Segments; Files in Scope: {x.MdFilesCount}; Remaining: {x.RemainingFiles}; Segments/s: {segmentsPerSec:n2}");
}

var progress = new Progress<(int TokenizedSegments, int MdFilesCount, int RemainingFiles)>(ReportProgress);
Console.CancelKeyPress += (s, e) =>
{
    if (cts.IsCancellationRequested) 
    {
        return;
    }

    Console.WriteLine("Cancelling...");
    cts.Cancel();
    e.Cancel = true;
};
Console.WriteLine("Press Ctrl+C to stop embedding gracefully...");

async Task EmbedderTask(string name, string path) 
{
    if (cts.IsCancellationRequested) 
    {
        return;
    }
    
    taskName = name;
    await embedder.EmbedMarkdownFilesAsync(path, progress, cts.Token);
}

await EmbedderTask("dotnet-docs", "D:/docs/dotnet-docs/docs/");
await EmbedderTask("aspnetcore", "D:/docs/aspnetcore/aspnetcore/");
await EmbedderTask("ef", "D:/docs/ef/entity-framework/");
await EmbedderTask("npgsql", "D:/docs/npgsql/conceptual/");
