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
        _sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider();
        _session = new InferenceSession(ModelOnnxPath, _sessionOptions);
        using var sr = File.OpenText(ModelVocabPath);
        _bertTokenizer.LoadVocabulary(sr, convertInputToLowercase: true);
    }

    public static IEnumerable<(float Similarity, Embedding Embedding)> EnumerateSimilarities(List<Embedding> corpus, float[] queryVector)
    {

    }

    public async Task<float[]> Embed(string input)
    {

    }
    
    // ToDo: implement IDisposable
}