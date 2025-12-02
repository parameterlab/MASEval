# User

In many real-world applications, Multi-Agent Systems (MAS) are designed to interact with human users to accomplish tasks. To effectively benchmark such systems, it is crucial to have a standardized way to simulate these interactions. The `User` class in MASEval provides this capability by acting as a programmable, LLM-driven user that can engage with the MAS in a realistic manner.

The User is initialized with a persona and a scenario, both of which are typically defined within a Task. This tight integration allows for dynamic and context-aware simulations. For example, a Task might generate a random birthdate for the user. This birthdate is then passed to both the `User` and the `Evaluator`. The User will use this information in its conversation with the MAS, and the `Evaluator` will check if the MAS correctly processes and remembers this information. This mechanism enables the creation of sophisticated and reliable benchmarks that can assess the interactive capabilities of a MAS.

::: maseval.core.user.User

## Interfaces

Some integrations provide convenience user/tool implementations for specific agent frameworks. For example:

::: maseval.interface.agents.smolagents.SmolAgentUser

::: maseval.interface.agents.langgraph.LangGraphUser

::: maseval.interface.agents.llamaindex.LlamaIndexUser
