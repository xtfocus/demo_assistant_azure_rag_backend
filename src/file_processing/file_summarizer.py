import random
from typing import Any, Dict, List

from openai import AsyncAzureOpenAI

from src.file_processing.pdf_parsing import FileImage


class FileSummarizer:
    def __init__(self, client: AsyncAzureOpenAI, config: Any, prompt: str):
        self.client = client
        self.config = config
        self.prompt = prompt
        self.max_samples = 5  # How many text and image items to sample (each)

    def _sample_items(self, items: List[str], max_samples: int) -> List[str]:
        """
        Sample up to max_samples items from the input list.
        If len(items) <= max_samples, returns all items.
        """
        if len(items) <= max_samples:
            return items
        if len(items) == 0:
            return items
        return [items[0]] + random.sample(items[1:], (max_samples - 1))

    def _create_message_content(
        self, images: List[str], texts: List[str]
    ) -> List[Dict]:
        """
        Create the message content for the API call.
        """
        content: List[Dict] = [{"type": "text", "text": self.prompt}]

        # Add sampled images
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image.image_base64}"
                    },
                }
            )

        # Add sampled texts
        if texts:
            combined_text = "\n\n".join(texts)
            content.append({"type": "text", "text": combined_text})

        return content

    async def run(
        self, texts: List[str], images: List[FileImage], temperature: float = None
    ) -> str:
        """
        Run the summarization process with sampling and validation.

        Args:
            images: List of base64 encoded image strings
            texts: List of text strings to summarize
            temperature: Optional temperature parameter for the API call

        Returns:
            str: Summarized content from the API response

        Raises:
            TypeError: If inputs are not of correct type
            ValueError: If no valid inputs are provided
        """

        # Sample inputs if necessary
        sampled_images = self._sample_items(images, self.max_samples)
        sampled_texts = self._sample_items(texts, self.max_samples)

        # Set temperature
        if temperature is None:
            temperature = self.config.temperature

        # Create API call content
        message_content = self._create_message_content(sampled_images, sampled_texts)

        # Make API call
        response = await self.client.chat.completions.create(
            model=self.config.MODEL_DEPLOYMENT,
            temperature=temperature,
            messages=[{"role": "user", "content": message_content}],
        )

        return response.choices[0].message.content
