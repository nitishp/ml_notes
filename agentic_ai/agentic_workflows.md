## Intro to Agentic Workflows

* The core skill here differentiating good agentic workflows is really just systematic error analysis and evals
* People normally use LLMs in a zero-shot approach
  * Think "Write an essay about topic X (but do no research / editing)"
  * Agentic AI is basically breaking this down into multiple LLM calls with smaller tasks
    * NOTE: Decomposing and breaking these down into tasks is a complex art in itself
      * It's more useful to think of this in terms of what building blocks you can use
      * This is part of the core workflow for how agents can be built
* There are degrees to how autonomous AI agents can be:
  * Less autonomous: Steps are predefined in a program. Most of the AI usage is in text generation
  * Highly automonous: AI makes decisions on what tools to call. AI can even make tools in highly advanced cases
* Benefits of Agentic AI
  * It improves performance on benchmarks. Adding agentic workflows on top of GPT 3.5 produced an **even bigger** jump in coding benchmarks compared to the non-agentic jump from 3.5 to 4
  * It allows parallelization and is faster
  * It's modular and you can swap out components used in the workflow (i.e. tools or LLMs)
* Agentic AI workflows are easier to build when there are clear step-by-step processes where there's a plan to follow. 
  * The modality of the input also matters. LLMs operate much better on text
* Evaluating agentic AI
  * Best practice is to build the agent first, then find examples where it's not satisfactory, and then work on how to improve or reduce these errors
    * Use evals for this for large scale changes
      * Can be objective (such as evaluating using code) or subjective (like using LLM as a judge)
      * Evals don't need to just be end-to-end, they can also be at the component level
      * Examine traces to perform systemic error analysis
* Agentic design patterns
  * Reflection: Ask the LLM to reflect on it's output or use tools to evaluate it's own output
  * Tool use: Give LLMs functions that they can call to make progress on the task
  * Planning: LLM decides what sequence of steps it needs to take in order to carry out the task
  * Multi-agent workflow: Sub agents that combine together to take care of the overall complex task


## Reflection Design Pattern

* Reflection is a step after the initial generation of the output to ask it to improve it's own output
  * It's much more powerful with external feedback. E.g. for a coding agent, if you had "external feedback" that gave feedback about whether or not it even compiles
* Zero shot prompting is where you give a prompt with no specific examples in the prompt
  * Similarly one shot prompting is if you give one example in the prompt. Few shot is if you give a few examples in the prompt
* Reflection pattern has been proven through research papers to improve the performance on various tasks
* Tips for writing reflection prompts
  * Clearly indicate the reflection action what things to look for
  * Tell it what specific things to look for (e.g. look at the image generated, critique it and update the chart)
  * Bonus is if you can give it an external tool in order to improve it's outputs (e.g. run the code and evaluate it's correctness)
    * At Meta, we're calling this "Graders". Because this information is fed back into the LLM
  * Using a different LLM for reflection might make sense to offset any biases
* How to evaluate the impact of reflection? 
  * This is where evals come in
  * For objective answers:
    * Collect ground truth labels/answers. And then compare the different agentic workflows (with reflection and without) and see which model is more accurate
  * For subjective answers:
    * One approach is to use an LLM as a judge. I.e. ask the LLM which image is better
      * This has their own problems. LLMs are not reliable and have internal biases
    * You can get more consistent results by giving the LLM a rubric on how to grade the response

## Tool Use
* Tools are functions that LLMs can call to get more accurate / up to date information
  * LLMs choose when to call tools as they deem appropriate
* How LLMs used to request tools to be called:
  * You would write a prompt telling the LLM what "tools" / functions they had access to with some signifier like "FUNCTION" to indicate they wanted the tool called
  * The LLM would then output "FUNCTION: get_current_time()" if it wanted the tool called
  * You would then call the function programmatically by parsing the output of the LLM
  * Then with the result and conversation history, you'd feed the info back to the LLM
* How LLMs request tools to be called now:
  * It sort of works the same way, but now there's some better APIs so you don't need to manually parse the LLM output, and call it again
  * You essentially use an API that gives it a detailed JSON of what the tool / function does
    * An important distinction is that the API will let the LLM to call the function for you