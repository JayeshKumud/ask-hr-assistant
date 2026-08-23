"""
Eval: adapts our ChatGroq instance to DeepEval's model interface.

DeepEval's metrics (FaithfulnessMetric, etc.) default to calling OpenAI.
Subclassing DeepEvalBaseLLM and implementing its four required methods
lets those same metrics run on whatever LLM we hand them instead —
here, the project's existing ChatGroq instance, so the eval script
doesn't need a second API key or provider.
"""
from deepeval.models.base_model import DeepEvalBaseLLM


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """Wraps a LangChain ChatGroq instance for use as a DeepEval judge model."""

    def __init__(self, llm):
        self._llm = llm
        super().__init__(model=llm.model_name)

    def load_model(self):
        return self._llm

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        response = await self._llm.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return self._llm.model_name


if __name__ == "__main__":
    # Manual check with a fake LLM — no network or real API key needed,
    # just confirms the DeepEvalBaseLLM interface is implemented correctly.
    class FakeLLM:
        model_name = "fake-model-for-testing"

        def invoke(self, prompt):
            class _Response:
                content = f"fake response to: {prompt[:30]}"

            return _Response()

    wrapped = GroqDeepEvalLLM(FakeLLM())
    print("Model name:", wrapped.get_model_name())
    print("generate() output:", wrapped.generate("What is the leave policy?"))