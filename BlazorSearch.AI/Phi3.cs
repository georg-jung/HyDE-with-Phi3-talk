using System.Text;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace BlazorSearch.AI;

public class Phi3 
{
    const string ModelDir = @"D:\hf\Phi-3.5-mini-instruct-onnx\cuda\cuda-int4-awq-block-128";

    const string AlternativePromptTemplate = @"<|system|>
You write concise technical documentation articles covering a topic and it's context.
NEVER write more than 100 words.
Skip introductory stuff, target an expert audience.
Naming key points is sufficient.
Don't make something up if you're unsure.
A short answer IS REQUIRED.
<|end|>
<|user|>Please write .NET documentation covering the following topic.
Topic: {0}<|end|>
<|assistant|>";

    const string PromptTemplate = @"<|system|>
You support a user with his search in .NET programming docs. His prompts might be full questions but could
also be in the style of google search queries. Do not create one definitive answer, but instead write five possibly helpful 
answers. What you create does not need to be a direct answer to a question, but it is also encouraged if you write 
about related or on-topic content, that is interesting for somebody who is interested in the user-prompted topic and wants 
to explore the field further.
RULES:
Focus on topics that might be covered in .NET docs.
Skip introductory stuff and assume your audience already has advanced knowledge.
Don't repeat yourself. Conciseness is key.
LAWS:
Create EXACTLY five distinct answers.
You MUST be very concise.
Answers MUST be english.
You MUST NOT write more than 100 words in total.
<|end|>
<|user|>Topic: {0}<|end|>
<|assistant|>";

    private readonly Model model;
    private readonly Tokenizer tokenizer;

    public Phi3() 
    {

    }

    public void GenerateDocs(string topic, StringBuilder output)
    {
        output.Append($"Represent this sentence for searching relevant passages: {topic}");
    }

    // ToDo: implement IDisposable
}