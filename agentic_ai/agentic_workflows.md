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


