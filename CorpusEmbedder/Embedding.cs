using System.Diagnostics.CodeAnalysis;

namespace CorpusEmbedder;

public record class Embedding
{
    public required EmbeddingKey Key { get; init; }

    public required float[] Vector { get; init; }
}

public readonly record struct EmbeddingKey
{
    public required string FilePath { get; init; }

    public required int Offset { get; init; }

    public required int? LastTokenizedWordStartIndex { get; init; }
}
