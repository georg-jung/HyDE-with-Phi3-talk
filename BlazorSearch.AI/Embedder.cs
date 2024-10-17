using System.Numerics.Tensors;
using FastBertTokenizer;
using Microsoft.ML.OnnxRuntime;

namespace BlazorSearch.AI;

public class Embedder
{
    const int BatchSize = 1;
    const int MaxTokens = 512;
    const int EmbeddingDimensions = 384;
    const string ModelOnnxPath = "D:/hf/bge-small-en-v1.5/model.onnx";
    const string ModelVocabPath = "D:/hf/bge-small-en-v1.5/vocab.txt";

    private readonly BertTokenizer _bertTokenizer = new();
    private readonly SessionOptions _sessionOptions;
    private readonly InferenceSession _session;

    public Embedder() 
    {

    }

    public static IEnumerable<(float Similarity, Embedding Embedding)> EnumerateSimilarities(List<Embedding> corpus, float[] queryVector)
    {

    }

    public async Task<List<(float Similarity, Embedding Embedding)>> GetSearchResults(List<Embedding> corpus, string query, int results = 32) 
    {

    }

    public async Task<float[]> Embed(string input)
    {

    }
    
    // ToDo: implement IDisposable
}