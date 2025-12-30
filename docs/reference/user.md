# User

In many real-world applications, Multi-Agent Systems (MAS) are designed to interact with human users to accomplish tasks. To effectively benchmark such systems, it is crucial to have a standardized way to simulate these interactions. The `User` class in MASEval provides this capability by acting as a programmable, LLM-driven user that can engage with the MAS in a realistic manner.

The User is initialized with a persona and a scenario, both of which are typically defined within a Task. This tight integration allows for dynamic and context-aware simulations. For example, a Task might generate a random birthdate for the user. This birthdate is then passed to both the `User` and the `Evaluator`. The User will use this information in its conversation with the MAS, and the `Evaluator` will check if the MAS correctly processes and remembers this information. This mechanism enables the creation of sophisticated and reliable benchmarks that can assess the interactive capabilities of a MAS.

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/core/user.py){ .md-source-file }

::: maseval.core.user.User

::: maseval.core.user.AgenticUser

## Interfaces

Some integrations provide convenience user/tool implementations for specific agent frameworks. For example:

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/agents/smolagents.py){ .md-source-file }

::: maseval.interface.agents.smolagents.SmolAgentUser

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/agents/langgraph.py){ .md-source-file }

::: maseval.interface.agents.langgraph.LangGraphUser

[:material-github: View source](https://github.com/parameterlab/maseval/blob/main/maseval/interface/agents/llamaindex.py){ .md-source-file }

::: maseval.interface.agents.llamaindex.LlamaIndexUser
