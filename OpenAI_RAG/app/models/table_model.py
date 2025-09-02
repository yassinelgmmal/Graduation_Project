import logging
import asyncio
import pandas as pd
from io import StringIO
from typing import Optional, Dict, Any, List
from openai import AsyncAzureOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

class TableModel:
    """
    Wrapper for table processing and summarization using Azure OpenAI models.
    """
    
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key_credits_account,
            api_version=settings.azure_openai_api_version_credits_account,
            azure_endpoint=settings.azure_openai_endpoint_credits_account
        )
        self.deployment_name = settings.table_deployment_name

    async def summarize(
        self,
        table_data: str,
        table_context: str = "",
        max_length: int = 512
    ) -> str:
        """
        Generate a summary for the given table using Azure OpenAI models.

        Args:
            table_data: String representation of the table (CSV, HTML, or structured text)
            table_context: Additional context about the table
            max_length: Maximum tokens in summary

        Returns:
            Generated summary string
        """
        try:
            prompt = f"""Analyze and summarize the following scientific table. Focus on:
            - Key trends and patterns in the data
            - Statistical significance of findings
            - Comparison between different groups/conditions
            - Notable values or outliers
            
            {f"Context: {table_context}" if table_context else ""}
            
            Table data:
            {table_data}
            
            Summary:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes and summarizes scientific tables and data."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_length
            )

            summary = response.choices[0].message.content.strip()
            print(f"Generated summary: {summary}")
            print(f"Summary: {response.choices[0].message.content}")
            logger.info(f"Generated table summary of length: {len(summary)}")
            return summary

        except Exception as e:
            logger.error(f"Error summarizing table: {e}")
            return f"Error summarizing table: {str(e)}"
            
    async def extract_key_findings(self, table_data: str, num_findings: int = 5) -> List[str]:
        """
        Extract key findings from table data using Azure OpenAI.
        
        Args:
            table_data: String representation of the table
            num_findings: Number of key findings to extract
            
        Returns:
            List of key findings
        """
        try:
            prompt = f"""Extract {num_findings} key findings from this scientific table. 
            Focus on the most important statistical results, trends, or comparisons.
            Format your response as a numbered list.
            
            Table data:
            {table_data}
            
            Key findings:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key findings from scientific tables."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=600
            )

            content = response.choices[0].message.content.strip()
            
            # Parse the numbered list into separate findings
            findings = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and clean up
                    clean_finding = line
                    if line[0].isdigit():
                        clean_finding = line.split('.', 1)[-1].strip()
                    elif line.startswith('-') or line.startswith('•'):
                        clean_finding = line[1:].strip()
                    
                    if clean_finding:
                        findings.append(clean_finding)

            logger.info(f"Extracted {len(findings)} key findings from table")
            return findings[:num_findings]

        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return []

    async def analyze_table_structure(self, table_data: str) -> Dict[str, Any]:
        """
        Analyze the structure and content of a table using Azure OpenAI.
        
        Args:
            table_data: String representation of the table
            
        Returns:
            Dictionary containing structure analysis
        """
        try:
            prompt = f"""Analyze the structure of this scientific table. Provide your response in the following format:

            Table Type: [comparison/results/statistics/demographic/experimental/other]
            Columns: [list the main column headers]
            Rows: [approximate number of data rows]
            Data Types: [numerical/categorical/mixed]
            Statistical Measures: [means/percentages/p-values/confidence_intervals/other]
            
            Table data:
            {table_data}
            
            Analysis:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes the structure of scientific tables."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=400
            )

            content = response.choices[0].message.content.strip()
            
            # Parse the response
            analysis = {
                "table_type": "unknown",
                "columns": [],
                "rows": 0,
                "data_types": "unknown",
                "statistical_measures": [],
                "raw_analysis": content,
                "status": "success"
            }

            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('Table Type:'):
                    analysis["table_type"] = line.split(':', 1)[1].strip()
                elif line.startswith('Columns:'):
                    columns_text = line.split(':', 1)[1].strip()
                    analysis["columns"] = [col.strip() for col in columns_text.split(',')]
                elif line.startswith('Rows:'):
                    try:
                        rows_text = line.split(':', 1)[1].strip()
                        analysis["rows"] = int(''.join(filter(str.isdigit, rows_text)))
                    except:
                        analysis["rows"] = 0
                elif line.startswith('Data Types:'):
                    analysis["data_types"] = line.split(':', 1)[1].strip()
                elif line.startswith('Statistical Measures:'):
                    measures_text = line.split(':', 1)[1].strip()
                    analysis["statistical_measures"] = [measure.strip() for measure in measures_text.split(',')]

            logger.info(f"Analyzed table structure: {analysis['table_type']}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing table structure: {e}")
            return {
                "table_type": "unknown",
                "columns": [],
                "rows": 0,
                "data_types": "unknown",
                "statistical_measures": [],
                "raw_analysis": "",
                "status": "error",
                "error": str(e)
            }

    async def compare_table_values(self, table_data: str, comparison_focus: str = "") -> str:
        """
        Generate comparisons between values in the table.
        
        Args:
            table_data: String representation of the table
            comparison_focus: Specific aspect to focus on for comparison
            
        Returns:
            Comparison analysis string
        """
        try:
            prompt = f"""Compare the values in this scientific table and highlight the most significant differences or similarities.
            {f"Focus particularly on: {comparison_focus}" if comparison_focus else ""}
            
            Provide statistical comparisons where possible (percentages, ratios, significance levels).
            
            Table data:
            {table_data}
            
            Comparison analysis:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that performs comparative analysis of scientific data."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=600
            )

            comparison = response.choices[0].message.content.strip()
            logger.info(f"Generated comparison analysis for table")
            return comparison

        except Exception as e:
            logger.error(f"Error comparing table values: {e}")
            return f"Error performing comparison: {str(e)}"

    async def format_table_for_analysis(self, raw_table: str) -> str:
        """
        Clean and format table data for better analysis.
        
        Args:
            raw_table: Raw table string (could be HTML, CSV, or other format)
            
        Returns:
            Cleaned and formatted table string
        """
        try:
            # Try to parse as CSV first
            if ',' in raw_table or '\t' in raw_table:
                try:
                    # Try CSV parsing
                    if ',' in raw_table:
                        df = pd.read_csv(StringIO(raw_table))
                    else:
                        df = pd.read_csv(StringIO(raw_table), sep='\t')
                    
                    # Convert back to a clean format
                    return df.to_string(index=False)
                except:
                    pass
            
            # If CSV parsing fails, clean up the raw text
            lines = raw_table.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('<') and not line.startswith('='):
                    # Remove excessive whitespace and normalize separators
                    cleaned_line = ' '.join(line.split())
                    cleaned_lines.append(cleaned_line)
            
            return '\n'.join(cleaned_lines)

        except Exception as e:
            logger.error(f"Error formatting table: {e}")
            return raw_table

    async def generate_table_caption(self, table_data: str, context: str = "") -> str:
        """
        Generate a scientific caption for a table.
        
        Args:
            table_data: String representation of the table
            context: Additional context about the table
            
        Returns:
            Generated caption string
        """
        try:
            prompt = f"""Generate a formal scientific caption for this table. The caption should:
            - Briefly describe what the table shows
            - Mention the key variables or measurements
            - Include any important statistical information
            - Follow scientific publication standards
            
            {f"Context: {context}" if context else ""}
            
            Table data:
            {table_data}
            
            Caption:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that writes scientific table captions following academic standards."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=200
            )

            caption = response.choices[0].message.content.strip()
            logger.info(f"Generated caption for table")
            return caption

        except Exception as e:
            logger.error(f"Error generating table caption: {e}")
            return f"Error generating caption: {str(e)}"

    async def comprehensive_table_analysis(self, table_data: str, context: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a table combining multiple methods.
        
        Args:
            table_data: String representation of the table
            context: Additional context about the table
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            # Format the table for better analysis
            formatted_table = await self.format_table_for_analysis(table_data)
            
            # Run multiple analyses concurrently
            summary_task = self.summarize_table(formatted_table, context)
            findings_task = self.extract_key_findings(formatted_table)
            structure_task = self.analyze_table_structure(formatted_table)
            comparison_task = self.compare_table_values(formatted_table)
            caption_task = self.generate_table_caption(formatted_table, context)

            summary, findings, structure, comparison, caption = await asyncio.gather(
                summary_task, findings_task, structure_task, comparison_task, caption_task
            )

            return {
                "summary": summary,
                "key_findings": findings,
                "structure_analysis": structure,
                "comparison_analysis": comparison,
                "suggested_caption": caption,
                "formatted_table": formatted_table,
                "analysis_status": "success"
            }

        except Exception as e:
            logger.error(f"Error in comprehensive table analysis: {e}")
            return {
                "summary": "Error generating summary",
                "key_findings": [],
                "structure_analysis": {
                    "table_type": "unknown",
                    "status": "error"
                },
                "comparison_analysis": "Error performing comparison",
                "suggested_caption": "Error generating caption",
                "formatted_table": table_data,
                "analysis_status": "error",
                "error": str(e)
            }