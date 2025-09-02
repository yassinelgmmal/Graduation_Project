import logging
import asyncio
from typing import Optional, List
from openai import AsyncAzureOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

class TextModel:
    """
    Wrapper for text processing using Azure OpenAI models for summarization and text generation.
    """
    
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key_credits_account,
            api_version=settings.azure_openai_api_version_credits_account,
            azure_endpoint=settings.azure_openai_endpoint_credits_account
        )
        self.deployment_name = settings.text_deployment_name

    async def summarize(
        self, 
        text: str, 
        prompt: str = "Summarize the following scientific text, focusing on key findings, methods, and conclusions:",
        max_length: int = 512
    ) -> str:
        """
        Generate a summary for the given text using Azure OpenAI models.

        Args:
            text: input text to summarize.
            prompt: custom prompt for summarization (used as context).
            max_length: maximum tokens in summary.

        Returns:
            Generated summary string.
        """
        try:
            # Analyze the length of the text
            text_length = len(text)
            logger.info(f"Summarizing text of length: {text_length}")
            
            # If text is very long, we need to be more strategic
            if text_length > 25000:
                logger.info("Text is very long, using chunked summarization")
                return await self._chunked_summarization(text, prompt, max_length)
            
            # Prepare a better system prompt
            system_prompt = """You are an expert scientific summarizer who creates clear, accurate, and comprehensive summaries 
            of scientific papers and technical documents. Your summaries should:
            
            1. Maintain technical accuracy and precision
            2. Capture the key methodologies, findings, and conclusions
            3. Include important technical details and metrics
            4. Be well-structured with logical flow
            5. Be comprehensive yet concise
            
            When summarizing, pay special attention to research methodology, experimental results, 
            key tables/figures, and the main contributions of the work.
            """
              # Prepare the full prompt
            full_prompt = f"{prompt}\n\nText to summarize:\n{text}\n\nSummary:"
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_completion_tokens=max_length
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Generated summary of length: {len(summary)}")
            
            # Check if the summary is empty and retry with different parameters
            if not summary:
                logger.warning("Received empty summary, retrying with different parameters")
                response = await self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": "Summarize the following text concisely."},
                        {"role": "user", "content": f"Please summarize this text:\n\n{text}"}
                    ],
                    max_completion_tokens=max_length
                )
                summary = response.choices[0].message.content.strip()
                logger.info(f"Retry generated summary of length: {len(summary)}")
                
                # If still empty, provide a basic fallback
                if not summary:
                    logger.error("Still received empty summary after retry")
                    return "Summary could not be generated for this content. The document may contain specialized content or formatting that is difficult to process."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            # Return a fallback summary
            return f"Error generating summary: {str(e)}"
    
    async def _chunked_summarization(self, text: str, prompt: str, max_length: int) -> str:
        """
        Handle summarization of very long texts by chunking.
        
        Args:
            text: Long text to summarize
            prompt: The summarization prompt
            max_length: Maximum length of final summary
            
        Returns:
            Generated summary string
        """
        # Split text into reasonable chunks
        chunk_size = 10000  # Characters per chunk
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        
        logger.info(f"Split text into {len(chunks)} chunks for summarization")
        
        # Generate intermediate summaries for each chunk
        intermediate_summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_prompt = f"Summarize this section of a scientific document (part {i+1} of {len(chunks)}):"
                response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert scientific summarizer. Create a concise summary of this document section."},
                    {"role": "user", "content": f"{chunk_prompt}\n\n{chunk}"}
                ],
                max_completion_tokens=300  # Shorter summaries for intermediate chunks
                )
                
                chunk_summary = response.choices[0].message.content.strip()
                intermediate_summaries.append(chunk_summary)
                logger.info(f"Generated intermediate summary {i+1}/{len(chunks)}")
                
            except Exception as e:
                logger.error(f"Error in chunk {i+1} summarization: {e}")
                intermediate_summaries.append(f"[Error summarizing chunk {i+1}]")
        
        # Combine intermediate summaries
        combined_intermediate = "\n\n".join([f"Part {i+1} Summary: {summary}" 
                                          for i, summary in enumerate(intermediate_summaries)])
        
        # Generate final summary from intermediate summaries
        try:
            final_prompt = f"""Generate a cohesive final summary of this scientific document based on these section summaries:

{combined_intermediate}

{prompt}"""            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an expert scientific summarizer. Create a comprehensive and cohesive summary from these section summaries."},
                    {"role": "user", "content": final_prompt}
                ],
                max_completion_tokens=max_length
            )
            
            final_summary = response.choices[0].message.content.strip()
            logger.info(f"Generated final summary of length: {len(final_summary)}")
            return final_summary
            
        except Exception as e:
            logger.error(f"Error in final summarization: {e}")
            # Fallback to just joining intermediate summaries
            if intermediate_summaries:
                return "Executive Summary:\n\n" + "\n\n".join(intermediate_summaries)
            
            # Ultimate fallback if everything else fails
            words = text.split()
            if len(words) > 50:
                return " ".join(words[:50]) + "..."
            return text
            
    async def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from scientific text using Azure OpenAI.
        """
        try:
            prompt = f"""Extract {num_points} key points from the following scientific text. 
            Format your response as a numbered list, with each point on a new line.
            Focus on the most important findings, methods, or conclusions.
            
            Text:
            {text}
            Key points:"""
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key points from scientific texts."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the numbered list into separate points
            points = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and clean up
                    clean_point = line
                    if line[0].isdigit():
                        clean_point = line.split('.', 1)[-1].strip()
                    elif line.startswith('-') or line.startswith('•'):
                        clean_point = line[1:].strip()
                    
                    if clean_point:
                        points.append(clean_point)
            
            logger.info(f"Extracted {len(points)} key points")
            return points[:num_points]  # Ensure we don't exceed requested number
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []

    async def classify_content(self, text: str) -> dict:
        """
        Classify the type and domain of scientific content using Azure OpenAI.
        """
        try:
            prompt = f"""Analyze the following scientific text and classify it. Provide your response in the following format:

            Domain: [Biology/Physics/Computer Science/Chemistry/Medicine/Mathematics/Other]
            Content Type: [Abstract/Introduction/Methods/Results/Discussion/Conclusion/Unknown]
            Confidence: [High/Medium/Low]
            
            Text to analyze:
            {text}
            Classification:"""
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies scientific texts by domain and content type."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the response
            analysis = {
                "domain": "General",
                "content_type": "Unknown",
                "confidence": "Low",
                "text_length": len(text),
                "processing_status": "success"
            }
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('Domain:'):
                    analysis["domain"] = line.split(':', 1)[1].strip()
                elif line.startswith('Content Type:'):
                    analysis["content_type"] = line.split(':', 1)[1].strip()
                elif line.startswith('Confidence:'):
                    analysis["confidence"] = line.split(':', 1)[1].strip()
            
            logger.info(f"Classified content as: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error classifying content: {e}")
            return {
                "domain": "Unknown",
                "content_type": "Unknown", 
                "confidence": "Low",
                "text_length": len(text),
                "processing_status": "error",
                "error": str(e)
            }    
    async def generate_text(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate text completion using Azure OpenAI models.
        
        Args:
            prompt: The input prompt for text generation
            max_length: Maximum length of generated text
            
        Returns:
            Generated text string
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates scientific text based on prompts."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_length
            )
            
            generated = response.choices[0].message.content.strip()
            logger.info(f"Generated text of length: {len(generated)}")
            return generated
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"

    async def analyze_scientific_text(self, text: str) -> dict:
        """
        Comprehensive analysis of scientific text combining multiple methods.
        
        Args:
            text: Input scientific text
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Run multiple analyses concurrently
            summary_task = self.summarize(text)
            key_points_task = self.extract_key_points(text)
            classification_task = self.classify_content(text)
            
            summary, key_points, classification = await asyncio.gather(
                summary_task, key_points_task, classification_task
            )
            
            return {
                "summary": summary,
                "key_points": key_points,
                "classification": classification,
                "word_count": len(text.split()),
                "char_count": len(text),
                "analysis_status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "summary": "Error generating summary",
                "key_points": [],
                "classification": {
                    "domain": "Unknown",
                    "content_type": "Unknown",
                    "confidence": "Low"
                },
                "word_count": len(text.split()) if text else 0,
                "char_count": len(text) if text else 0,
                "analysis_status": "error",
                "error": str(e)
            }