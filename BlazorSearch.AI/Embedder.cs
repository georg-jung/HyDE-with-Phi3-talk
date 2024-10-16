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

    public async Task<float[]> Embed(string input)
    {
        var inputIds = new long[MaxTokens];
        var attentionMask = new long[MaxTokens];
        var tokenTypeIds = new long[MaxTokens];
        _bertTokenizer.Encode(input, inputIds, attentionMask, tokenTypeIds);
        return await Embed(inputIds, attentionMask, tokenTypeIds);
    }

    public async Task<float[]> Embed(long[] inputIds, long[] attentionMask, long[] tokenTypeIds)
    {
        using var inputIdsTensor = OrtValue.CreateTensorValueFromMemory(inputIds, [BatchSize, MaxTokens]);
        using var attentionMaskTensor = OrtValue.CreateTensorValueFromMemory(attentionMask, [BatchSize, MaxTokens]);
        using var tokenTypeIdsTensor = OrtValue.CreateTensorValueFromMemory(tokenTypeIds, [BatchSize, MaxTokens]);
        using var output = OrtValue.CreateTensorValueFromMemory(new float[MaxTokens * EmbeddingDimensions], [BatchSize, MaxTokens, EmbeddingDimensions]);

        await _session.RunAsync(
                new RunOptions() { },
                ["input_ids", "attention_mask", "token_type_ids"],
                [inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor],
                ["last_hidden_state"],
                [output]);

        return [.. output.GetTensorDataAsSpan<float>().Slice(0, EmbeddingDimensions)];
    }
    
    // ToDo: implement IDisposable
}