import logging
import asyncio
import base64
from typing import Optional, Dict, Any
from pathlib import Path
from PIL import Image
from io import BytesIO
from openai import AsyncAzureOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

class FigureModel:
    """
    Wrapper for figure analysis and captioning using Azure OpenAI models.
    Falls back to text-based analysis if vision capabilities are not available.
    """
    
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key_credits_account,
            api_version=settings.azure_openai_api_version_credits_account,
            azure_endpoint=settings.azure_openai_endpoint_credits_account
        )
        self.deployment_name = settings.figure_deployment_name
        self.vision_available = True  # Default to assuming vision is available, will fallback if not

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string for Azure OpenAI."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            return ""

    def _encode_pil_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        try:
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding PIL image: {e}")
            return ""

    def _extract_info_from_filename(self, filename: str) -> str:
        """Extract information from figure filename."""
        try:
            # Common figure-related keywords
            figure_keywords = {
                'plot': 'data plot or chart',
                'graph': 'graphical representation',
                'chart': 'data chart',
                'scatter': 'scatter plot',
                'bar': 'bar chart',
                'line': 'line graph',
                'hist': 'histogram',
                'box': 'box plot',
                'heat': 'heatmap',
                'flow': 'flowchart',
                'diagram': 'diagram or schematic',
                'fig': 'figure',
                'image': 'image or photograph',
                'schematic': 'schematic diagram',
                'timeline': 'timeline visualization',
                'network': 'network diagram'
            }
            
            filename_lower = filename.lower()
            detected_types = []
            
            for keyword, description in figure_keywords.items():
                if keyword in filename_lower:
                    detected_types.append(description)
            
            if detected_types:
                return f"Likely contains: {', '.join(detected_types)}"
            else:
                return "General scientific figure"
                
        except Exception:
            return ""

    async def analyze_figure(
        self, 
        image_path: str, 
        context: str = "", 
        prompt: str = "Analyze this scientific figure and describe its content, data trends, and key insights."
    ) -> Dict[str, Any]:
        """
        Analyze a figure using Azure OpenAI models.
        Tries vision-based analysis first, falls back to text-based analysis if needed.
        
        Args:
            image_path: Path to the image file
            context: Additional context about the figure
            prompt: Custom prompt for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # First, try vision-based analysis if available
            if self.vision_available:
                result = await self._analyze_with_vision(image_path, context, prompt)
                if result.get("status") == "success":
                    return result
                else:
                    logger.warning("Vision analysis failed, falling back to text-based analysis")
                    self.vision_available = False
            
            # Fallback to text-based analysis
            return await self._analyze_without_vision(image_path, context, prompt)
            
        except Exception as e:
            logger.error(f"Error analyzing figure {image_path}: {e}")
            return {
                "caption": "Error analyzing figure",
                "analysis": f"Error: {str(e)}",
                "insights": [],
                "image_path": image_path,
                "status": "error",
                "error": str(e)
            }

    async def _analyze_with_vision(
        self, 
        image_path: str, 
        context: str, 
        prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze figure using vision capabilities (GPT-4V or GPT-4o-mini with vision).
        """
        try:
            # Encode the image
            base64_image = self._encode_image(image_path)
            if not base64_image:
                return {"status": "error", "error": "Could not encode image"}

            # Prepare the prompt
            full_prompt = f"{prompt}"
            if context:
                full_prompt += f"\n\nContext: {context}"

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes scientific figures and provides detailed descriptions and insights."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )

            analysis_text = response.choices[0].message.content.strip()
            
            # Extract insights from the analysis
            insights = self._extract_insights(analysis_text)
            
            result = {
                "caption": analysis_text,
                "analysis": analysis_text,
                "insights": insights,
                "image_path": image_path,
                "status": "success",
                "analysis_method": "vision"
            }
            
            logger.info(f"Successfully analyzed figure with vision: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Vision analysis failed for {image_path}: {e}")
            return {"status": "error", "error": str(e)}

    async def _analyze_without_vision(
        self, 
        image_path: str, 
        context: str, 
        prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze figure without vision capabilities using filename and context.
        """
        try:
            # Extract information from filename and path
            filename = Path(image_path).name
            file_info = self._extract_info_from_filename(filename)
            
            # Create a text-based analysis prompt
            analysis_prompt = f"""Analyze a scientific figure based on the following information:

            Filename: {filename}
            File path: {image_path}
            {f"Context: {context}" if context else ""}
            {f"File info extracted: {file_info}" if file_info else ""}
            
            Based on this information, provide an analysis that includes:
            1. Likely content type (graph, chart, diagram, etc.)
            2. Possible data being presented
            3. Key insights that might be shown
            4. Scientific relevance
            
            Original prompt: {prompt}
            
            Provide a comprehensive analysis as if you could see the image:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes scientific figures. Even without seeing the image, provide insights based on available context."
                    },
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=600
            )

            analysis_text = response.choices[0].message.content.strip()
            
            # Add disclaimer about text-based analysis
            disclaimer = "\n\n[Note: This analysis is based on filename and context only, as direct image analysis is not available.]"
            analysis_text += disclaimer
            
            # Extract insights from the analysis
            insights = self._extract_insights(analysis_text)
            
            result = {
                "caption": analysis_text,
                "analysis": analysis_text,
                "insights": insights,
                "image_path": image_path,
                "status": "success",
                "analysis_method": "text_based",
                "disclaimer": "Analysis based on filename and context only"
            }
            
            logger.info(f"Successfully analyzed figure without vision: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Text-based analysis failed for {image_path}: {e}")
            return {
                "caption": "Error analyzing figure",
                "analysis": f"Error: {str(e)}",
                "insights": [],
                "image_path": image_path,
                "status": "error",
                "error": str(e),
                "analysis_method": "failed"
            }

    async def caption(
        self, 
        image_path: str, 
        context: str = "",
        style: str = "scientific"
    ) -> str:
        """
        Generate a caption for a scientific figure.
        
        Args:
            image_path: Path to the image file
            context: Additional context about the figure
            style: Caption style (scientific, brief, detailed)
            
        Returns:
            Generated caption string
        """
        try:
            # Try vision-based captioning first
            if self.vision_available:
                caption = await self._generate_caption_with_vision(image_path, context, style)
                if caption and not caption.startswith("Error"):
                    return caption
                else:
                    logger.warning("Vision captioning failed, falling back to text-based captioning")
                    self.vision_available = False
            
            # Fallback to text-based captioning
            return await self._generate_caption_without_vision(image_path, context, style)
            
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return f"Error generating caption: {str(e)}"

    async def _generate_caption_with_vision(
        self, 
        image_path: str, 
        context: str, 
        style: str
    ) -> str:
        """Generate caption using vision capabilities."""
        try:
            base64_image = self._encode_image(image_path)
            if not base64_image:
                return "Error: Could not process image"

            if style == "scientific":
                prompt = "Generate a formal scientific caption for this figure. Include what type of data is shown, key trends, and statistical information if visible."
            elif style == "brief":
                prompt = "Generate a brief, concise caption describing what this figure shows."
            else:  # detailed
                prompt = "Generate a detailed caption explaining this figure, including methodology, data interpretation, and significance."

            if context:
                prompt += f"\n\nAdditional context: {context}"

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates scientific figure captions."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

            caption = response.choices[0].message.content.strip()
            print(f"Generated summary: {caption}")
            print(f"Summary: {response.choices[0].message.content}")
            logger.info(f"Generated vision-based caption for {image_path}")
            return caption
            
        except Exception as e:
            logger.error(f"Vision captioning failed for {image_path}: {e}")
            return f"Error: {str(e)}"

    async def _generate_caption_without_vision(
        self, 
        image_path: str, 
        context: str, 
        style: str
    ) -> str:
        """Generate caption without vision capabilities."""
        try:
            filename = Path(image_path).name
            file_info = self._extract_info_from_filename(filename)
            
            if style == "scientific":
                style_instruction = "Generate a formal scientific caption"
            elif style == "brief":
                style_instruction = "Generate a brief, concise caption"
            else:  # detailed
                style_instruction = "Generate a detailed caption"

            prompt = f"""{style_instruction} for a scientific figure based on the following information:

            Filename: {filename}
            {f"File info: {file_info}" if file_info else ""}
            {f"Context: {context}" if context else ""}
            
            Provide a professional caption that describes what the figure likely contains and its scientific relevance."""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates scientific figure captions based on available context."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )

            caption = response.choices[0].message.content.strip()
            caption += " [Caption generated from filename and context]"
            
            logger.info(f"Generated text-based caption for {image_path}")
            return caption
            
        except Exception as e:
            logger.error(f"Text-based captioning failed for {image_path}: {e}")
            return f"Error generating caption: {str(e)}"

    async def extract_text_from_figure(self, image_path: str) -> str:
        """
        Extract any text content from a figure.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text string or analysis based on available capabilities
        """
        try:
            if self.vision_available:
                return await self._extract_text_with_vision(image_path)
            else:
                return await self._extract_text_without_vision(image_path)
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return f"Error extracting text: {str(e)}"

    async def _extract_text_with_vision(self, image_path: str) -> str:
        """Extract text using vision capabilities."""
        try:
            base64_image = self._encode_image(image_path)
            if not base64_image:
                return "Error: Could not process image"

            prompt = """Extract all text visible in this image, including:
            - Axis labels and titles
            - Legend text
            - Data labels and values
            - Any annotations or captions
            
            Format the extracted text clearly and maintain any hierarchical structure."""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts text from scientific figures."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )

            extracted_text = response.choices[0].message.content.strip()
            logger.info(f"Extracted text from {image_path} using vision")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Vision text extraction failed for {image_path}: {e}")
            self.vision_available = False
            return await self._extract_text_without_vision(image_path)

    async def _extract_text_without_vision(self, image_path: str) -> str:
        """Extract text information without vision capabilities."""
        try:
            filename = Path(image_path).name
            file_info = self._extract_info_from_filename(filename)
            
            return f"""Text extraction not available without vision capabilities.
            
            Based on filename analysis:
            - Filename: {filename}
            - {file_info if file_info else 'No specific information could be extracted from filename'}
            
            [Note: Actual text extraction requires vision-enabled model]"""
            
        except Exception as e:
            logger.error(f"Text extraction fallback failed for {image_path}: {e}")
            return f"Error in text extraction: {str(e)}"

    async def classify_figure_type(self, image_path: str) -> Dict[str, Any]:
        """
        Classify the type of scientific figure.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with classification results
        """
        try:
            if self.vision_available:
                return await self._classify_with_vision(image_path)
            else:
                return await self._classify_without_vision(image_path)
            
        except Exception as e:
            logger.error(f"Error classifying figure {image_path}: {e}")
            return {
                "type": "unknown",
                "confidence": "low",
                "description": "",
                "data_type": "unknown",
                "image_path": image_path,
                "status": "error",
                "error": str(e)
            }

    async def _classify_with_vision(self, image_path: str) -> Dict[str, Any]:
        """Classify figure using vision capabilities."""
        try:
            base64_image = self._encode_image(image_path)
            if not base64_image:
                return {"type": "unknown", "confidence": "low", "error": "Could not process image"}

            prompt = """Classify this scientific figure. Provide your response in the following format:
            
            Figure Type: [line_plot/bar_chart/scatter_plot/histogram/heatmap/pie_chart/box_plot/flowchart/diagram/microscopy/other]
            Confidence: [high/medium/low]
            Description: [brief description of what the figure shows]
            Data Type: [quantitative/qualitative/mixed]"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that classifies scientific figures by type and content."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )

            content = response.choices[0].message.content.strip()
            
            # Parse the response
            classification = {
                "type": "unknown",
                "confidence": "low",
                "description": "",
                "data_type": "unknown",
                "image_path": image_path,
                "status": "success",
                "analysis_method": "vision"
            }
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('Figure Type:'):
                    classification["type"] = line.split(':', 1)[1].strip()
                elif line.startswith('Confidence:'):
                    classification["confidence"] = line.split(':', 1)[1].strip()
                elif line.startswith('Description:'):
                    classification["description"] = line.split(':', 1)[1].strip()
                elif line.startswith('Data Type:'):
                    classification["data_type"] = line.split(':', 1)[1].strip()
            
            logger.info(f"Classified figure {image_path} with vision: {classification}")
            return classification
            
        except Exception as e:
            logger.error(f"Vision classification failed for {image_path}: {e}")
            self.vision_available = False
            return await self._classify_without_vision(image_path)

    async def _classify_without_vision(self, image_path: str) -> Dict[str, Any]:
        """Classify figure without vision capabilities."""
        try:
            filename = Path(image_path).name
            file_info = self._extract_info_from_filename(filename)
            
            classification = {
                "type": "unknown",
                "confidence": "low",
                "description": f"Classification based on filename: {filename}",
                "data_type": "unknown",
                "image_path": image_path,
                "status": "success",
                "analysis_method": "text_based"
            }
            
            # Try to classify based on filename
            filename_lower = filename.lower()
            if any(word in filename_lower for word in ['plot', 'graph', 'chart']):
                classification["type"] = "chart"
                classification["confidence"] = "medium"
            elif any(word in filename_lower for word in ['diagram', 'schematic']):
                classification["type"] = "diagram"
                classification["confidence"] = "medium"
            elif any(word in filename_lower for word in ['flow', 'flowchart']):
                classification["type"] = "flowchart"
                classification["confidence"] = "medium"
                
            if file_info:
                classification["description"] += f" - {file_info}"
            
            classification["description"] += " [Classification based on filename only]"
            
            logger.info(f"Classified figure {image_path} without vision: {classification}")
            return classification
            
        except Exception as e:
            logger.error(f"Text-based classification failed for {image_path}: {e}")
            return {
                "type": "unknown",
                "confidence": "low",
                "description": "",
                "data_type": "unknown",
                "image_path": image_path,
                "status": "error",
                "error": str(e),
                "analysis_method": "failed"
            }

    def _extract_insights(self, analysis_text: str) -> list[str]:
        """Extract key insights from analysis text."""
        try:
            # Simple heuristic to extract insights
            insights = []
            sentences = analysis_text.split('.')
            
            insight_keywords = [
                'shows', 'indicates', 'demonstrates', 'reveals', 'suggests',
                'trend', 'pattern', 'correlation', 'significant', 'increase',
                'decrease', 'comparison', 'relationship'
            ]
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Filter out very short sentences
                    for keyword in insight_keywords:
                        if keyword.lower() in sentence.lower():
                            insights.append(sentence + '.')
                            break
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []

    async def summarize_multiple_figures(self, image_paths: list[str], context: str = "") -> Dict[str, Any]:
        """
        Analyze multiple figures and provide a summary.
        
        Args:
            image_paths: List of paths to image files
            context: Additional context about the figures
            
        Returns:
            Dictionary containing summary of all figures
        """
        try:
            # Analyze each figure individually
            figure_analyses = []
            
            for image_path in image_paths:
                analysis = await self.analyze_figure(image_path, context)
                figure_analyses.append(analysis)
            
            # Create a comprehensive summary
            all_insights = []
            successful_analyses = [a for a in figure_analyses if a.get('status') == 'success']
            
            for analysis in successful_analyses:
                all_insights.extend(analysis.get('insights', []))
            
            summary = {
                "total_figures": len(image_paths),
                "successful_analyses": len(successful_analyses),
                "failed_analyses": len(image_paths) - len(successful_analyses),
                "all_insights": all_insights,
                "individual_analyses": figure_analyses,
                "vision_capability": self.vision_available,
                "status": "success" if successful_analyses else "error"
            }
            
            logger.info(f"Summarized {len(image_paths)} figures")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing multiple figures: {e}")
            return {
                "total_figures": len(image_paths),
                "successful_analyses": 0,
                "failed_analyses": len(image_paths),
                "all_insights": [],
                "individual_analyses": [],
                "vision_capability": self.vision_available,
                "status": "error",
                "error": str(e)
            }