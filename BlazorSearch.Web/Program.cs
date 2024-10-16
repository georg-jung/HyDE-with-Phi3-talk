using System.IO.Compression;
using System.Text.Json;
using BlazorSearch.AI;
using BlazorSearch.Web.Components;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

var embs = new List<Embedding>();
await using var fs = File.OpenRead("D:/docs/BAAI_bge-small-en-v1.5_embeddings.jsonl.br");
await using var brotliStream = new BrotliStream(fs, CompressionMode.Decompress);
using var sr = new StreamReader(brotliStream);
while (!sr.EndOfStream && await sr.ReadLineAsync() is string line) 
{
    var emb = JsonSerializer.Deserialize<Embedding>(line);
    embs.Add(emb!);
}

builder.Services.AddSingleton(embs);
builder.Services.AddSingleton<Embedder>();
builder.Services.AddSingleton<Phi3>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
