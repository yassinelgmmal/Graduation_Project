class SummarizationStrategy:
    """
    Base class for different summarization strategies
    """
    def summarize(self, content):
        raise NotImplementedError("Each summarization strategy must implement this method")

class TextSummarizer(SummarizationStrategy):
    """
    Summarizer for textual content using PEGASUS API
    """
    def __init__(self, api_client):
        self.api_client = api_client
        
    def summarize(self, text, max_length=500, min_length=50):
        """
        Summarize text content
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            
        Returns:
            str: Generated summary
        """
        if not text or len(text.strip()) < min_length:
            return ""
            
        return self.api_client.summarize_text(text, max_length, min_length)

class TableSummarizer(SummarizationStrategy):
    """
    Summarizer for table content using Qwen Table API
    """
    def __init__(self, api_client):
        self.api_client = api_client
        
    def summarize(self, table_image_path):
        """
        Summarize table content
        
        Args:
            table_image_path (str): Path to table image
            
        Returns:
            str: Generated table summary
        """
        if not table_image_path:
            return ""
            
        return self.api_client.summarize_table(table_image_path)

class FigureSummarizer(SummarizationStrategy):
    """
    Summarizer for figure content using Azure OpenAI API
    """
    def __init__(self, api_client):
        self.api_client = api_client
        
    def summarize(self, figure_image_path):
        """
        Summarize figure content
        
        Args:
            figure_image_path (str): Path to figure image
            
        Returns:
            str: Generated figure description
        """
        if not figure_image_path:
            return ""
            
        return self.api_client.summarize_figure(figure_image_path)

class SummarizationFactory:
    """
    Factory for creating appropriate summarizer based on content type
    """
    def __init__(self, api_client):
        self.api_client = api_client
        
    def get_summarizer(self, content_type):
        """
        Get the appropriate summarizer for the content type
        
        Args:
            content_type (str): Type of content ('text', 'table', or 'figure')
            
        Returns:
            SummarizationStrategy: Appropriate summarizer instance
        """
        if content_type == "text":
            return TextSummarizer(self.api_client)
        elif content_type == "table":
            return TableSummarizer(self.api_client)
        elif content_type == "figure":
            return FigureSummarizer(self.api_client)
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

class MultimodalSummarizer:
    """
    Coordinator for multimodal summarization
    """
    def __init__(self, api_client):
        self.factory = SummarizationFactory(api_client)
        
    def summarize_paper_content(self, paper_data):
        """
        Generate summaries for all content types in a paper
        
        Args:
            paper_data (dict): Extracted paper data
            
        Returns:
            dict: Paper data with summaries added
        """
        # Create a deep copy to avoid modifying the original
        result = paper_data.copy()
        
        # Summarize text sections
        text_summarizer = self.factory.get_summarizer("text")
        for section in result.get("text_sections", []):
            section["summary"] = text_summarizer.summarize(section.get("text", ""))
            
        # Summarize tables
        table_summarizer = self.factory.get_summarizer("table")
        for table in result.get("tables", []):
            table["summary"] = table_summarizer.summarize(table.get("image_path", ""))
            
        # Summarize figures
        figure_summarizer = self.factory.get_summarizer("figure")
        for figure in result.get("figures", []):
            figure["summary"] = figure_summarizer.summarize(figure.get("image_path", ""))
            
        return result
