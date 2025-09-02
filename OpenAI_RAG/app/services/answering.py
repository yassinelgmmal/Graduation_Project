from typing import List, Tuple, Dict, Optional
import logging
import asyncio
from openai import AsyncAzureOpenAI

from app.services.retrieval import retrieve_documents_with_scores
from app.config import settings

logger = logging.getLogger(__name__)

# Create a singleton instance of AnsweringService
_answering_service = None

async def get_answering_service():
    """
    Get or create the singleton AnsweringService instance.
    """
    global _answering_service
    if _answering_service is None:
        _answering_service = AnsweringService()
    return _answering_service

async def answer_query(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.3,  # Lowered from 0.5 to 0.3 to get more results
    document_id: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Answer a query by retrieving relevant documents and generating an answer.
    
    Args:
        query: The user's question
        top_k: Number of documents to retrieve
        score_threshold: Minimum relevance score for documents
        document_id: Optional document ID to filter results
        
    Returns:
        Tuple of (answer, sources)
    """
    service = await get_answering_service()
    
    # Prepare filter for document_id if provided
    filter_kwargs = {"document_id": document_id} if document_id else None
    
    print(f"Answering service query: {query}")
    result = await service.answer_with_retrieval(
        question=query,
        top_k=top_k,
        score_threshold=score_threshold,
        filter_kwargs=filter_kwargs
    )
    print(f"Answering service result: {result}")
    return result.get("answer", ""), result.get("sources", [])

async def generate_follow_up_questions(
    query: str,
    answer: str,
    sources: List[Dict],
    num_questions: int = 3
) -> List[str]:
    """
    Generate follow-up questions based on a query and its answer.
    
    Args:
        query: The original query
        answer: The generated answer
        sources: The sources used to generate the answer
        num_questions: Number of follow-up questions to generate
        
    Returns:
        List of follow-up questions
    """
    service = await get_answering_service()
    
    # Convert sources to context documents format
    context_documents = [{"content": src.get("content", ""), "metadata": src} for src in sources]
    
    return await service.generate_follow_up_questions(
        question=query,
        answer=answer,
        context_documents=context_documents,
        num_questions=num_questions
    )

class AnsweringService:
    """
    Service for generating answers using Azure OpenAI models.
    """
    
    def __init__(self):
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key_credits_account,
            api_version=settings.azure_openai_api_version_credits_account,
            azure_endpoint=settings.azure_openai_endpoint_credits_account
        )
        self.deployment_name = settings.text_deployment_name

    async def generate_answer(
        self,
        question: str,
        context_documents: List[Dict],
        max_tokens: int = 1000
    ) -> Dict:
        """
        Generate an answer to the question using the provided context documents.
        
        Args:
            question: The user's question
            context_documents: List of retrieved documents with content and metadata
            max_tokens: Maximum tokens in the response
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Prepare context from documents
            context_text = self._prepare_context(context_documents)
              # Create the prompt
            system_prompt = "You are a helpful assistant that answers scientific questions based on provided research documents. Always be accurate and cite the source material when possible."
            
            # Special case handling for certain terms
            query_lower = question.lower()
            if any(term in query_lower for term in ["dataset", "corpus", "benchmark"]):
                system_prompt += " Be particularly detailed when explaining datasets, benchmarks, or evaluation metrics mentioned in the documents."
            
            prompt = f"""Based on the following scientific documents, answer the question accurately and comprehensively.

Context:
{context_text}

Question: {question}

Instructions:
- Provide a detailed, accurate answer based on the given context
- Cite specific information from the documents when possible
- If the context doesn't contain enough information to answer fully, state this clearly
- Focus on scientific accuracy and precision
- Include relevant details, statistics, or findings from the documents

Answer:"""            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )

            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(question, context_documents, answer)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "sources": [doc.get("metadata", {}) for doc in context_documents],
                "context_used": len(context_documents),
                "status": "success"
            }
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            print(f"Generated answer: {result['answer']}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while generating an answer: {str(e)}",                "confidence": 0.0,
                "sources": [],
                "context_used": 0,
                "status": "error",
                "error": str(e)
            }

    async def answer_with_retrieval(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
        max_tokens: int = 800,
        filter_kwargs: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Answer a question by first retrieving relevant documents and then generating an answer.
        
        Args:
            question: The user's question
            top_k: Number of documents to retrieve
            score_threshold: Minimum relevance score for documents
            max_tokens: Maximum tokens in the response
            filter_kwargs: Optional metadata filters, e.g., {"document_id": "uuid"}
            
        Returns:
            Dictionary containing the answer and retrieval metadata
        """
        try:
            # Normalize the query to improve search
            normalized_query = question.strip().rstrip("?")
            if len(normalized_query) < 3:
                logger.warning(f"Query too short: {question}")
            
            logger.info(f"Processing question: {question}")
            logger.info(f"Using score threshold: {score_threshold}")
            
            # Retrieve relevant documents
            retrieved_docs = await retrieve_documents_with_scores(
                query=normalized_query,
                top_k=top_k,
                score_threshold=score_threshold,
                filter_kwargs=filter_kwargs
            )
            
            if not retrieved_docs:
                logger.warning(f"No documents found for query: {question}")
                
                # Try a more lenient search with lower threshold
                if score_threshold > 0.0:
                    logger.info(f"Retrying with lower threshold: 0.0")
                    retrieved_docs = await retrieve_documents_with_scores(
                        query=normalized_query,
                        top_k=top_k,
                        score_threshold=0.0,
                        filter_kwargs=filter_kwargs
                    )
                
                if not retrieved_docs:
                    return {
                        "answer": "I couldn't find relevant documents to answer your question. Please try rephrasing your question or ask about a different topic.",
                        "confidence": 0.0,
                        "sources": [],
                        "retrieved_documents": 0,
                        "status": "no_documents"
                    }
            
            # Generate answer using retrieved documents
            result = await self.generate_answer(
                question=question,
                context_documents=retrieved_docs,
                max_tokens=max_tokens
            )
            
            # Add retrieval metadata
            result["retrieved_documents"] = len(retrieved_docs)
            result["retrieval_scores"] = [doc.get("score", 0.0) for doc in retrieved_docs]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in answer_with_retrieval: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "retrieved_documents": 0,
                "status": "error",
                "error": str(e)
            }

    async def generate_follow_up_questions(
        self,
        question: str,
        answer: str,
        context_documents: List[Dict],
        num_questions: int = 3
    ) -> List[str]:
        """
        Generate follow-up questions based on the original question and answer.
        
        Args:
            question: Original question
            answer: Generated answer
            context_documents: Context documents used
            num_questions: Number of follow-up questions to generate
            
        Returns:
            List of follow-up questions
        """
        try:
            context_text = self._prepare_context(context_documents, max_length=1000)
            prompt = f"""Based on the following question, answer, and scientific context, generate {num_questions} thoughtful follow-up questions that would help deepen understanding of the topic.

Original Question: {question}

Answer: {answer}

Context: {context_text}

Generate follow-up questions that:
- Explore related aspects of the topic
- Ask for more specific details
- Connect to broader scientific concepts
- Are answerable from the available context

Follow-up questions:"""
            
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates insightful follow-up questions for scientific discussions."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1000
            )

            content = response.choices[0].message.content.strip()
            logger.debug(f"Raw follow-up questions response: {content}")
            
            # Parse the questions
            questions = []
            
            # First try to parse numbered or bulleted questions
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets and clean up
                    clean_question = line
                    if line[0].isdigit():
                        clean_question = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    elif line.startswith('-') or line.startswith('•'):
                        clean_question = line[1:].strip()
                    
                    if clean_question and '?' in clean_question:
                        questions.append(clean_question)
            
            # If no questions found, try to find any sentences with question marks
            if not questions:
                for line in content.split('\n'):
                    line = line.strip()
                    if '?' in line:
                        # Split by question marks and reform questions
                        parts = line.split('?')
                        for i in range(len(parts) - 1):  # Last part after ? is not a question
                            question = parts[i].strip() + '?'
                            if len(question) > 10:  # Avoid very short questions
                                questions.append(question)
            
            # If still no questions found, generate default questions
            if not questions and answer:
                # Extract key topics from the answer
                topics = self._extract_topics(answer)
                for topic in topics[:num_questions]:
                    questions.append(f"Can you elaborate more on {topic}?")
            
            # Ensure we have at least one question
            if not questions:
                questions = [
                    "Can you provide more details about this topic?",
                    "What are the practical applications of this research?",
                    "How does this compare to other similar studies in the field?"
                ]
            
            logger.info(f"Generated {len(questions)} follow-up questions")
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}")
            return []

    async def explain_scientific_concept(
        self,
        concept: str,
        context_documents: List[Dict],
        complexity_level: str = "intermediate"
    ) -> str:
        """
        Provide an explanation of a scientific concept based on context documents.
        
        Args:
            concept: Scientific concept to explain
            context_documents: Relevant documents for context
            complexity_level: Level of explanation (basic, intermediate, advanced)
            
        Returns:
            Explanation string
        """
        try:
            context_text = self._prepare_context(context_documents)
            
            if complexity_level == "basic":
                level_instruction = "Explain in simple terms that a general audience can understand."
            elif complexity_level == "advanced":
                level_instruction = "Provide a detailed, technical explanation suitable for experts."
            else:  # intermediate
                level_instruction = "Provide a comprehensive explanation suitable for students and professionals."

            prompt = f"""Based on the following scientific documents, explain the concept of "{concept}".

{level_instruction}

Context:
{context_text}

Please provide:
1. A clear definition of the concept
2. Key principles or mechanisms involved
3. Relevant examples or applications from the documents
4. Significance in the field

Explanation:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that explains scientific concepts clearly and accurately."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=800
            )

            explanation = response.choices[0].message.content.strip()
            logger.info(f"Generated explanation for concept: {concept}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining concept {concept}: {e}")
            return f"Error generating explanation: {str(e)}"

    def _prepare_context(self, documents: List[Dict], max_length: int = 2000) -> str:
        """
        Prepare context text from retrieved documents.
        
        Args:
            documents: List of document dictionaries
            max_length: Maximum length of context text
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Create a formatted document entry
            doc_entry = f"Document {i+1}"
            if metadata.get("source"):
                doc_entry += f" (Source: {metadata['source']})"
            doc_entry += f":\n{content}\n"
            
            if current_length + len(doc_entry) > max_length:
                break
                
            context_parts.append(doc_entry)
            current_length += len(doc_entry)
        
        return "\n".join(context_parts)

    def _calculate_confidence(
        self,
        question: str,
        context_documents: List[Dict],
        answer: str
    ) -> float:
        """
        Calculate confidence score for the generated answer.
        
        Args:
            question: Original question
            context_documents: Context documents used
            answer: Generated answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Simple heuristic-based confidence calculation
            confidence = 0.5  # Base confidence
            
            # Increase confidence based on number of quality documents
            doc_count = len(context_documents)
            if doc_count >= 3:
                confidence += 0.2
            elif doc_count >= 2:
                confidence += 0.1
            
            # Increase confidence based on document scores
            avg_score = sum(doc.get("score", 0.0) for doc in context_documents) / max(doc_count, 1)
            confidence += avg_score * 0.3
            
            # Decrease confidence if answer is very short or contains error indicators
            if len(answer) < 50:
                confidence -= 0.2
            elif "I don't know" in answer or "cannot answer" in answer:
                confidence -= 0.3
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Default moderate confidence

    async def explain_methodology(
        self,
        document_id: str,
        aspect: str = "general"
    ) -> str:
        """
        Explain the methodology used in a specific document.
        
        Args:
            document_id: ID of the document to explain methodology for
            aspect: Specific aspect of methodology to focus on
            
        Returns:
            Explanation of the methodology
        """
        from app.services.retrieval import retrieve_documents_with_scores
        
        try:
            # Get all documents chunks for this document
            logger.info(f"Retrieving methodology for document: {document_id}")
            document_chunks = await retrieve_documents_with_scores(
                query="methodology experiment procedure method analysis data collection",  # Focus on methodology terms
                top_k=20,  # Get more chunks to ensure we have comprehensive data
                filter_kwargs={"document_id": document_id},
                score_threshold=0.0  # Get all chunks for this document
            )
            
            if not document_chunks:
                logger.warning(f"No document chunks found for document ID: {document_id}")
                return "Document not found or no content available for methodology analysis."
            
            # Log the number of chunks found
            logger.info(f"Found {len(document_chunks)} chunks for document ID: {document_id}")
            
            # Create context documents in the expected format
            # Sort by score to prioritize most relevant chunks
            document_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Convert to the format expected by generate_answer
            context_documents = [{"content": doc.get("content", ""), "metadata": doc.get("metadata", {})} 
                               for doc in document_chunks[:10]]  # Use top 10 chunks
            
            # Create a specialized prompt for methodology extraction
            methodology_prompt = """
            Extract and explain the methodology from the provided scientific document chunks.
            
            Focus on identifying:
            1. Research design (experimental, observational, qualitative, quantitative, mixed methods)
            2. Data collection methods (surveys, interviews, instruments, measurements)
            3. Sampling approach and participant selection
            4. Analysis techniques (statistical methods, qualitative analysis)
            5. Tools, software or equipment used
            6. Validation or verification methods
            
            Even if the methodology is not explicitly stated, infer it from descriptions of procedures, 
            experiments, analyses, or technical approaches mentioned in the text.
            
            If specific methodology terms are mentioned (e.g., "ablation study", "cross-validation", 
            "t-test", "thematic analysis"), highlight and explain these.
            """
            
            # Create query based on aspect
            if aspect == "general":
                query = "What methodology was used in this research? Explain in detail."
            else:
                query = f"Explain the {aspect} methodology used in this research. Be specific and detailed."
                
                # Add aspect-specific instructions
                if "data" in aspect.lower():
                    methodology_prompt += "\nFocus especially on data collection methods, data sources, and data processing."
                elif "analysis" in aspect.lower():
                    methodology_prompt += "\nFocus especially on analytical methods, statistical techniques, and analysis procedures."
                elif "experiment" in aspect.lower():
                    methodology_prompt += "\nFocus especially on experimental design, controls, variables, and experimental procedures."
            
            # Generate answer with specialized prompt
            result = await self.generate_specialized_answer(
                question=query,
                context_documents=context_documents,
                system_instruction=methodology_prompt
            )
            
            return result.get("answer", "No methodology information could be extracted.")
            
        except Exception as e:
            logger.error(f"Error explaining methodology: {e}")
            return f"Error explaining methodology: {str(e)}"
            
    async def generate_specialized_answer(
        self,
        question: str,
        context_documents: List[Dict],
        system_instruction: str,
        max_tokens: int = 1000
    ) -> Dict:
        """
        Generate a specialized answer with custom system instructions.
        
        Args:
            question: The user's question
            context_documents: List of retrieved documents with content and metadata
            system_instruction: Specialized system instructions
            max_tokens: Maximum tokens in the response
            
        Returns:
            Dictionary containing the answer and metadata
        """
        try:
            # Prepare context from documents
            context_text = self._prepare_context(context_documents, max_length=4000)  # Use more context
            
            # Create the prompt
            prompt = f"""Based on the following scientific document excerpts, answer the question thoroughly.

Context:
{context_text}

Question: {question}

Instructions:
- Extract all relevant methodological information from the context
- If methodological details are spread across different excerpts, synthesize them
- Be specific about research design, data collection, and analysis approaches
- If the methodology isn't explicitly stated, infer it from descriptions of work done
- Cite specific sections or pages when possible

Answer:"""

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": system_instruction
                    },
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=max_tokens
            )

            answer = response.choices[0].message.content.strip()
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(question, context_documents, answer)
            
            result = {
                "answer": answer,
                "confidence": confidence,
                "sources": [doc.get("metadata", {}) for doc in context_documents],
                "context_used": len(context_documents),
                "status": "success"
            }
            
            logger.info(f"Generated specialized answer for question: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error generating specialized answer: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while generating an answer: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "context_used": 0,
                "status": "error",
                "error": str(e)
            }    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text for generating follow-up questions.
        
        Args:
            text: The text to extract topics from
            
        Returns:
            List of key topics as strings
        """
        topics = []
        try:
            # Split into sentences
            sentences = text.split('.')
            
            # Extract important terms and phrases
            important_terms = []
            
            # Look for capitalized terms which often indicate important concepts
            import re
            capitalized_terms = re.findall(r'[A-Z][a-zA-Z]{2,}', text)
            important_terms.extend(capitalized_terms)
            
            # Look for terms in quotes which are often important
            quoted_terms = re.findall(r'"([^"]+)"', text)
            important_terms.extend(quoted_terms)
            
            # Extract noun phrases using basic heuristics
            for sentence in sentences:
                words = sentence.strip().split()
                if len(words) >= 3:
                    # Look for noun phrases of 2-3 words
                    for i in range(len(words) - 2):
                        # Check for phrases that might be important
                        potential_phrase = ' '.join(words[i:i+3])
                        if len(potential_phrase) > 5 and not any(word.lower() in [
                            'is', 'are', 'was', 'were', 'be', 'been', 'this', 'that', 
                            'these', 'those', 'their', 'other'
                        ] for word in words[i:i+3]):
                            important_terms.append(potential_phrase)
            
            # Add single important words that are likely topics
            for sentence in sentences:
                words = sentence.strip().split()
                for word in words:
                    if len(word) > 4 and word.lower() not in [
                        'about', 'their', 'these', 'those', 'which', 'where', 
                        'when', 'what', 'that', 'this'
                    ]:
                        important_terms.append(word)
            
            # De-duplicate and take top topics
            unique_terms = []
            for term in important_terms:
                term = term.strip()
                # Skip very short terms and common words
                if len(term) < 4 or term.lower() in [
                    'the', 'and', 'for', 'with', 'that', 'this', 'from', 'have', 'has'
                ]:
                    continue
                if term not in unique_terms:
                    unique_terms.append(term)
            
            # If we didn't find enough terms, use fallbacks
            if len(unique_terms) < 3:
                return ["this topic", "the methodology", "the findings", "the implications", "the research"]
                
            return unique_terms[:5]  # Return up to 5 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return ["this topic", "the methodology", "the findings", "the implications", "the research"]
                
            return unique_terms[:5]  # Return up to 5 topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return ["this topic", "the methodology", "the findings", "the implications", "the research"]

async def explain_methodology(document_id: str, aspect: str = "general") -> str:
    """
    Explain the methodology used in a specific document.
    
    Args:
        document_id: ID of the document to explain methodology for
        aspect: Specific aspect of methodology to focus on
        
    Returns:
        Explanation of the methodology
    """
    service = await get_answering_service()
    return await service.explain_methodology(document_id, aspect)
