import argparse
import os
import json
from src.paper_processor import ScientificPaperProcessor

def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(description="Scientific Paper Multimodal Summarization Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process paper command
    process_parser = subparsers.add_parser("process", help="Process a scientific paper")
    process_parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query about a paper")
    query_parser.add_argument("--paper_id", help="Paper ID to query")
    query_parser.add_argument("--query", required=True, help="Query text")
    
    # List papers command
    list_parser = subparsers.add_parser("list", help="List processed papers")
    
    # Get summary command
    summary_parser = subparsers.add_parser("summary", help="Get paper summary")
    summary_parser.add_argument("--paper_id", required=True, help="Paper ID")
    summary_parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = ScientificPaperProcessor()
    
    if args.command == "process":
        # Check if file exists
        if not os.path.exists(args.pdf):
            print(f"Error: PDF file not found: {args.pdf}")
            return
            
        print(f"Processing paper: {args.pdf}")
        paper_data = processor.process_pdf(args.pdf)
        
        if paper_data:
            print(f"Paper processed successfully. Paper ID: {paper_data['paper_id']}")
            print(f"Title: {paper_data['title']}")
            print(f"Data saved to: data/papers/{paper_data['paper_id']}.json")
        else:
            print("Failed to process paper.")
            
    elif args.command == "query":
        if args.paper_id:
            print(f"Querying paper {args.paper_id} with: {args.query}")
            results = processor.query_paper(args.paper_id, args.query)
        else:
            print(f"Querying all papers with: {args.query}")
            results = {
                "query": args.query,
                "relevant_chunks": []
            }
            relevant_chunks = processor.rag_manager.retrieve_relevant_chunks(args.query, k=10)
            
            for doc in relevant_chunks:
                results["relevant_chunks"].append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                
        if results and results.get("relevant_chunks"):
            print(f"Found {len(results['relevant_chunks'])} relevant chunks:")
            for i, chunk in enumerate(results["relevant_chunks"]):
                print(f"\n--- Chunk {i+1} ---")
                print(f"Source: {chunk['metadata'].get('source', 'unknown')}")
                print(f"Content: {chunk['content'][:200]}...")
        else:
            print("No relevant information found.")
            
    elif args.command == "list":
        print("Listing all processed papers:")
        papers_dir = "data/papers"
        
        if os.path.exists(papers_dir):
            for filename in os.listdir(papers_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(papers_dir, filename), 'r') as f:
                        paper_data = json.load(f)
                        print(f"ID: {paper_data['paper_id']}")
                        print(f"Title: {paper_data['title']}")
                        print(f"Authors: {', '.join(paper_data['authors'])}")
                        print("---")
        else:
            print("No papers found.")
            
    elif args.command == "summary":
        print(f"Getting summary for paper: {args.paper_id}")
        
        try:
            with open(f"data/papers/{args.paper_id}.json", 'r') as f:
                paper_data = json.load(f)
                
            summary = processor.generate_multimodal_summary(paper_data)
            
            print(f"Title: {summary['title']}")
            print(f"Authors: {', '.join(summary['authors'])}")
            print("\nOverall Summary:")
            print(summary['overall_summary'])
            
            print("\nClassifications:")
            for category in summary['classifications']:
                print(f"- {category}")
                
            print("\nText Sections:")
            for section in summary['text_sections']:
                print(f"\n{section['title']}:")
                print(section['summary'])
                
            print("\nTables:")
            for table in summary['tables']:
                print(f"\n{table['caption']}:")
                print(table['summary'])
                
            print("\nFigures:")
            for figure in summary['figures']:
                print(f"\n{figure['caption']}:")
                print(figure['description'])
                
            # Save to output file if specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f"\nSummary saved to {args.output}")
                
        except FileNotFoundError:
            print(f"Error: Paper with ID {args.paper_id} not found.")
        except Exception as e:
            print(f"Error retrieving summary: {str(e)}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
