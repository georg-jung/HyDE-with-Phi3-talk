using Markdig;
using Markdig.Extensions.Yaml;
using Markdig.Renderers;
using Markdig.Syntax;
using YamlDotNet.Core;
using YamlDotNet.Core.Events;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

public static class MarkdownHelper
{
    public static (string? Title, string Html) Parse(string markdown) 
    {
        var pipeline = new MarkdownPipelineBuilder()
            .UseYamlFrontMatter()
            .Build();

        using var writer = new StringWriter();
        var renderer = new HtmlRenderer(writer);
        pipeline.Setup(renderer);

        MarkdownDocument document = Markdown.Parse(markdown, pipeline);
        var yamlBlock = document.Descendants<YamlFrontMatterBlock>().FirstOrDefault();

        string? title = null;
        if (yamlBlock != null)
        {
            var yaml = markdown.Substring(yamlBlock.Span.Start, yamlBlock.Span.Length);
            using var yamlReader = new StringReader(yaml);
            var deserializer = new DeserializerBuilder()
                .WithCaseInsensitivePropertyMatching()
                .IgnoreUnmatchedProperties()
                .Build();
            var parser = new Parser(yamlReader);
            parser.Consume<StreamStart>();
            parser.Consume<DocumentStart>();
            var fm = deserializer.Deserialize<Frontmatter>(parser);
            parser.Consume<DocumentEnd>();
            
            title = fm?.Title;
        }

        renderer.Render(document);
        writer.Flush();
        string html = writer.ToString();
        return (title, html);
    }

    private class Frontmatter 
    {
        public string? Title { get; set; }
    }
}