using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using System.Threading.Channels;
using FastBertTokenizer;
using Microsoft.ML.OnnxRuntime;

namespace CorpusEmbedder;

public class Embedder : IDisposable
{
    private readonly SessionOptions _sessOpt;
    private readonly InferenceSession _session;
    private readonly BertTokenizer _tokenizer;
    private readonly Dictionary<EmbeddingKey, Embedding> _embeddings;
    private readonly string _embeddingsJsonlBrPath;
    private readonly int _embeddingDimensions;
    private readonly int _maxTokens;
    public readonly bool _sentenceEmbeddingModelLayout;

    private Embedder(SessionOptions sessOpt,
                     InferenceSession session,
                     BertTokenizer tokenizer,
                     Dictionary<EmbeddingKey, Embedding> embeddings,
                     string embeddingsJsonlBrPath,
                     int embeddingDimensions,
                     int maxTokens,
                     bool sentenceEmbeddingModelLayout)
    {
        _sessOpt = sessOpt;
        _session = session;
        _tokenizer = tokenizer;
        _embeddings = embeddings;
        _embeddingsJsonlBrPath = embeddingsJsonlBrPath;
        _embeddingDimensions = embeddingDimensions;
        _maxTokens = maxTokens;
        _sentenceEmbeddingModelLayout = sentenceEmbeddingModelLayout;
    }

    public static async Task<Embedder> CreateUaeLargeAsync()
    {
        const string ModelOnnxUrl = "https://huggingface.co/WhereIsAI/UAE-Large-V1/resolve/cf14327/onnx/model_quantized.onnx";
        const string ModelVocabUrl = "https://huggingface.co/WhereIsAI/UAE-Large-V1/resolve/cf14327/vocab.txt";
        const string EmbeddingsPath = "D:/docs/WhereIsAI_UAE-Large-V1_embeddings.jsonl.br";
        const string ModelDir = "D:/hf/WhereIsAI_UAE-Large-V1";
        return await CreateForModelAsync(ModelOnnxUrl, ModelVocabUrl, EmbeddingsPath, ModelDir, "model_quantized.onnx", "vocab.txt", embeddingDimensions: 1024, maxTokens: 512, sentenceEmbeddingModelLayout: false);
    }

    public static async Task<Embedder> CreateStella500mAsync()
    {
        const string ModelOnnxUrl = "https://huggingface.co/dunzhang/stella_en_400M_v5/resolve/1543163/onnx/model_quantized.onnx";
        const string ModelVocabUrl = "https://huggingface.co/dunzhang/stella_en_400M_v5/resolve/1543163/vocab.txt";
        const string EmbeddingsPath = "D:/docs/dunzhang_stella_en_400M_v5_embeddings.jsonl.br";
        const string ModelDir = "D:/hf/dunzhang_stella_en_400M_v5";
        return await CreateForModelAsync(ModelOnnxUrl, ModelVocabUrl, EmbeddingsPath, ModelDir, "model_quantized.onnx", "vocab.txt", embeddingDimensions: 1024, maxTokens: 8192, sentenceEmbeddingModelLayout: true);
    }

    public static async Task<Embedder> CreateBgeSmallEn1dot5Async()
    {
        const string ModelOnnxUrl = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7/onnx/model.onnx";
        const string ModelVocabUrl = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/5c38ec7/vocab.txt";
        const string EmbeddingsPath = "D:/docs/BAAI_bge-small-en-v1.5_embeddings.jsonl.br";
        const string ModelDir = "D:/hf/bge-small-en-v1.5";
        return await CreateForModelAsync(ModelOnnxUrl, ModelVocabUrl, EmbeddingsPath, ModelDir, "model.onnx", "vocab.txt", embeddingDimensions: 384, maxTokens: 512, sentenceEmbeddingModelLayout: false);
    }

    public static async Task<Embedder> CreateBgeBaseEn1dot5Async()
    {
        const string ModelName = "bge-base-en-v1.5";
        const string ModelOnnxUrl = $"https://huggingface.co/BAAI/{ModelName}/resolve/a5beb1e/onnx/model.onnx";
        const string ModelVocabUrl = $"https://huggingface.co/BAAI/{ModelName}/resolve/a5beb1e/vocab.txt";
        const string EmbeddingsPath = $"D:/docs/BAAI_{ModelName}_embeddings.jsonl.br";
        const string ModelDir = $"D:/hf/{ModelName}";
        return await CreateForModelAsync(ModelOnnxUrl, ModelVocabUrl, EmbeddingsPath, ModelDir, "model.onnx", "vocab.txt", embeddingDimensions: 768, maxTokens: 512, sentenceEmbeddingModelLayout: false);
    }

    public static async Task<Embedder> CreateForModelAsync(string modelOnnxUrl,
                                                           string modelVocabUrl,
                                                           string embeddingsPath,
                                                           string modelDir,
                                                           string modelOnnxName,
                                                           string vocabName,
                                                           int embeddingDimensions,
                                                           int maxTokens,
                                                           bool sentenceEmbeddingModelLayout)
    {
        string modelOnnxPath = Path.Combine(modelDir, modelOnnxName);
        string modelVocabPath = Path.Combine(modelDir, vocabName);

        if (!Directory.Exists(modelDir))
        {
            Directory.CreateDirectory(modelDir);
        }

        if (Path.GetDirectoryName(embeddingsPath) is string dirPath && !Directory.Exists(dirPath))
        {
            Directory.CreateDirectory(dirPath);
        }
        
        static async Task EnsureDownloaded(string url, string file) 
        {
            if (File.Exists(file)) 
            {
                return;
            }
            using var client = new HttpClient();
            await using var fs = File.Create(file);
            await using var httpStream = await client.GetStreamAsync(url);
            await httpStream.CopyToAsync(fs);
        }

        await Task.WhenAll(EnsureDownloaded(modelOnnxUrl, modelOnnxPath), EnsureDownloaded(modelVocabUrl, modelVocabPath));
        return await Embedder.CreateAsync(embeddingsPath, modelOnnxPath, modelVocabPath, useGpu: true, embeddingDimensions, maxTokens, sentenceEmbeddingModelLayout);
    }

    public static async Task<Embedder> CreateAsync(string embeddingsJsonlBrPath, string modelOnnx, string modelVocabTxt, bool useGpu, int embeddingDimensions, int maxTokens, bool sentenceEmbeddingModelLayout)
    {
        SessionOptions sessOpt = useGpu ? SessionOptions.MakeSessionOptionWithCudaProvider() : new SessionOptions();
        var session = new InferenceSession(modelOnnx, sessOpt);

        var tokenizer = new BertTokenizer();
        await tokenizer.LoadVocabularyAsync(modelVocabTxt, true);

        var embs = new Dictionary<EmbeddingKey, Embedding>();
        if (File.Exists(embeddingsJsonlBrPath)) 
        {
            await using var fs = File.OpenRead(embeddingsJsonlBrPath);
            await using var brotliStream = new BrotliStream(fs, CompressionMode.Decompress);
            using var sr = new StreamReader(brotliStream);
            while (!sr.EndOfStream && await sr.ReadLineAsync() is string line) 
            {
                var emb = JsonSerializer.Deserialize<Embedding>(line);
                embs.Add(emb!.Key, emb);
            }
        }

        return new Embedder(sessOpt, session, tokenizer, embs, embeddingsJsonlBrPath, embeddingDimensions, maxTokens, sentenceEmbeddingModelLayout);
    }

    public async Task EmbedMarkdownFilesAsync(string markdownDirectory, IProgress<(int TokenizedSegments, int MdFilesCount, int RemainingFiles)>? progress, CancellationToken cancellationToken)
    {
        const int BatchSize = 50;
        const int Stride = 16;
        using var tokenTypeIdsTensor = OrtValue.CreateTensorValueFromMemory(new long[BatchSize * _maxTokens], [BatchSize, _maxTokens]);
        using var output = OrtValue.CreateTensorValueFromMemory(
            new float[BatchSize * (_sentenceEmbeddingModelLayout ? 1 : _maxTokens) * _embeddingDimensions], 
            _sentenceEmbeddingModelLayout ? [BatchSize, _embeddingDimensions] : [BatchSize, _maxTokens, _embeddingDimensions]);

        HashSet<string> paths = [.. _embeddings.Keys.Select(k => k.FilePath)];
        Func<string, bool> predicate = path => !paths.Contains(path);
        var allMdFiles = Directory.EnumerateFiles(markdownDirectory, "*.md", SearchOption.AllDirectories).Count();

        var mdContentEnumerable = ReadMarkdownFiles(markdownDirectory, predicate);
        var tokenizedEnumerable = _tokenizer.CreateAsyncBatchEnumerator(mdContentEnumerable, _maxTokens, BatchSize, Stride);
        
        var cnt = 0;
        HashSet<string> filesSeen = [.. paths];
        await foreach (var batch in tokenizedEnumerable)
        {
            var inputIds = batch.InputIds;
            var attentionMask = batch.AttentionMask;

            using var iidsTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, inputIds, [BatchSize, _maxTokens]);
            using var attmTensor = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, attentionMask, [BatchSize, _maxTokens]);
            await _session.RunAsync(
                new RunOptions() { },
                ["input_ids", "attention_mask", "token_type_ids"],
                [iidsTensor, attmTensor, tokenTypeIdsTensor],
                [_sentenceEmbeddingModelLayout ? "sentence_embedding" : "last_hidden_state"],
                [output]);

            for (var i = 0; i < BatchSize; i++)
            {
                var corr = batch.OutputCorrelation.Span[i];
                if (!corr.HasValue)
                {
                    continue;
                }

                float[] embedding = [.. output.GetTensorDataAsSpan<float>().Slice(i * _maxTokens * _embeddingDimensions, _embeddingDimensions)];
                var embeddingEntity = new Embedding
                {
                    Key = new EmbeddingKey
                    {
                        FilePath = corr.Value.Key,
                        Offset = corr.Value.Offset,
                        LastTokenizedWordStartIndex = corr.Value.LastTokenizedWordStartIndex
                    },
                    Vector = embedding,
                };
                
                _embeddings[embeddingEntity.Key] = embeddingEntity;
                cnt++;
                filesSeen.Add(embeddingEntity.Key.FilePath);
                progress?.Report((cnt, allMdFiles, allMdFiles - filesSeen.Count));
            }

            if (cancellationToken.IsCancellationRequested) 
            {
                break;
            }
        }

        if (!cancellationToken.IsCancellationRequested)
        {
            Console.WriteLine("Successfully finished embedding!");
        }

        Console.WriteLine($"Writing to {_embeddingsJsonlBrPath}...");
        await using var fs = File.Create(_embeddingsJsonlBrPath);
        await using var brotliStream = new BrotliStream(fs, CompressionMode.Compress);
        using var sw = new StreamWriter(brotliStream);
        foreach (var (_, emb) in _embeddings) 
        {
            await sw.WriteLineAsync(JsonSerializer.Serialize(emb));
        }
        Console.WriteLine($"Finished writing.");
    }

    private static async IAsyncEnumerable<(string Path, string Content)> ReadMarkdownFiles(string markdownDirectory, Func<string, bool> filePathPredicate)
    {
        // Keep a cache of 10 already read files. This enables disk I/O and tokenization to happen in parallel.
        var channel = Channel.CreateBounded<(string Path, string Content)>(10);
        var fillChannelTask = Task.Run(() => FillChannelWithMarkdownFileContents(channel.Writer, markdownDirectory, filePathPredicate));

        await foreach (var x in channel.Reader.ReadAllAsync())
        {
            yield return x;
        }
        await fillChannelTask;
    }

    private static async Task FillChannelWithMarkdownFileContents(ChannelWriter<(string Path, string Content)> c, string markdownDirectory, Func<string, bool> filePathPredicate)
    {
        foreach (var file in Directory.EnumerateFiles(markdownDirectory, "*.md", SearchOption.AllDirectories).Where(filePathPredicate))
        {
            var content = await File.ReadAllTextAsync(file);
            await c.WriteAsync((file, content));
        }

        c.Complete();
    }

    public void Dispose()
    {
        _sessOpt.Dispose();
        _session.Dispose();
    }
}
