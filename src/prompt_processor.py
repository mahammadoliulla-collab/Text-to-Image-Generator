class PromptProcessor:

    def clean(self, prompt):

        prompt = prompt.strip()

        prompt = " ".join(prompt.split())

        return prompt