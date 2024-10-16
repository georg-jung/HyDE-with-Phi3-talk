using System.Threading.Channels;
using FastBertTokenizer;

namespace CorpusEmbedder;

internal static class BertTokenizerExtensions
{
    public static async IAsyncEnumerable<(string Path, int Offset, int? LastTokenizedWordStartIndex, Memory<long> InputIds, Memory<long> AttentionMask)>
        EnumerateEncodedMarkdownFiles(
            this BertTokenizer tokenizer,
            string markdownDirectory,
            int tokensPerInput,
            Func<string, bool> filePathPredicate)
    {
        var mdContentEnumerable = ReadMarkdownFiles(markdownDirectory, filePathPredicate);
        var tokenizedEnumerable = tokenizer.CreateAsyncBatchEnumerator(mdContentEnumerable, tokensPerInput, 1, 16);
        await foreach (var x in tokenizedEnumerable)
        {
            var corr = x.OutputCorrelation.Span[0]!.Value;
            yield return (corr.Key, corr.Offset, corr.LastTokenizedWordStartIndex, x.InputIds, x.AttentionMask);
        }
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
}
