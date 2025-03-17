import streamlit as st
import pandas as pd
import time
import openai
import elasticsearch
from elasticsearch import Elasticsearch
import time
from tabulate import tabulate
import os
from anthropic import Anthropic
import anthropic
import json

def retrieve_chunks_hybrid(es, open_api_key, index_name, query_text, top_k=100, final_k=10,
                            semantic_weight=0.7, bm25_weight=0.3, enable_hybrid = True):
    
    if semantic_weight + bm25_weight != 1.0:
        total = semantic_weight + bm25_weight
        semantic_weight /= total
        bm25_weight /= total

    openai.api_key = open_api_key
                                    
    response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[str(query_text)],  
            encoding_format="float"
        )
        
    query_vector = response.data[0].embedding


    semantic_hits = {}
    bm25_hits = {}

    semantic_body = {
        "size": top_k * 2,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        "_source": {"excludes": ["embedding"]}
    }
    
    try:
        semantic_response = es.search(index=index_name, body=semantic_body)
        for hit in semantic_response['hits']['hits']:
            semantic_hits[hit['_id']] = {
                'score': hit['_score'],
                'source': hit['_source'],
                'meta': {
                    'index': hit.get('_index'),
                    'doc_type': hit.get('_type'),
                    'doc_id': hit['_id']
                }
            }
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []

    # **BM25 Search (if enabled)**
    if enable_hybrid:
        bm25_body = {
            "size": top_k * 2,
            "query": {
                "match": {
                    "Description": {
                        "query": query_text,
                        "operator": "or"
                    }
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
        
        try:
            bm25_response = es.search(index=index_name, body=bm25_body)
            for hit in bm25_response['hits']['hits']:
                bm25_hits[hit['_id']] = {
                    'score': hit['_score'],
                    'source': hit['_source'],
                    'meta': {
                        'index': hit.get('_index'),
                        'doc_type': hit.get('_type'),
                        'doc_id': hit['_id']
                    }
                }
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

    def normalize_scores(scores_dict):
        if not scores_dict:
            return {}

        values = [item['score'] for item in scores_dict.values()]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        
        if max_val == min_val:
            return {doc_id: {
                'score': 0.5, 
                'source': item['source'],
                'meta': item.get('meta', {})
            } for doc_id, item in scores_dict.items()}
        
        return {doc_id: {
            'score': (item['score'] - min_val) / (max_val - min_val), 
            'source': item['source'],
            'meta': item.get('meta', {})
        } for doc_id, item in scores_dict.items()}
    
    
    normalized_semantic = normalize_scores(semantic_hits)
    normalized_bm25 = normalize_scores(bm25_hits)
    
    all_doc_ids = set(semantic_hits.keys()).union(bm25_hits.keys())
    
    combined_results = []
    
    for doc_id in all_doc_ids:
        original_semantic = semantic_hits.get(doc_id, {}).get('score', 0)
        original_bm25 = bm25_hits.get(doc_id, {}).get('score', 0)
        
        norm_semantic = normalized_semantic.get(doc_id, {}).get('score', 0)
        norm_bm25 = normalized_bm25.get(doc_id, {}).get('score', 0)
        
        blended_score = (semantic_weight * norm_semantic) + (bm25_weight * norm_bm25)
        
        # Preserve metadata from whichever search found the document
        source = semantic_hits.get(doc_id, {}).get('source') or bm25_hits.get(doc_id, {}).get('source')
        meta = semantic_hits.get(doc_id, {}).get('meta') or bm25_hits.get(doc_id, {}).get('meta') or {'doc_id': doc_id}
        
        if source:
            combined_results.append({
                'id': doc_id,
                'original_semantic': original_semantic,
                'original_bm25': original_bm25,
                'normalized_semantic': norm_semantic,
                'normalized_bm25': norm_bm25,
                'blended_score': blended_score,
                'source': source,
                'meta': meta
            })

    sorted_results = sorted(combined_results, key=lambda x: x['blended_score'], reverse=True)
    
    selected_results = sorted_results[:final_k]
    
    output = []
    for i, result in enumerate(selected_results, 1):
            
        source = result['source']
        norm_semantic = result['normalized_semantic']
        norm_bm25 = result['normalized_bm25']
        
        weighted_semantic = norm_semantic * semantic_weight
        weighted_bm25 = norm_bm25 * bm25_weight
        
        output.append({
            "rank": i,
            "original_semantic": round(result['original_semantic'], 4),
            "original_bm25": round(result['original_bm25'], 4),
            "normalized_semantic": round(norm_semantic, 4),
            "normalized_bm25": round(norm_bm25, 4),
            "blended_score": round(result['blended_score'], 4),
            "id": result['id'],  # Document ID
            "doc_id": result['meta'].get('doc_id', result['id']),  
            "restaurant": source.get('Restaurant_Name'),
            "restaurant_id": source.get('Restaurant_id'),  # Added Restaurant ID
            "yelp_url": source.get('Yelp_URL'),  # Added Yelp URL
            "address": source.get('Address'),  # Added Address
            "city": source.get('City'),  # Added City
            "zip": source.get('Zip'),  # Added Zip
            "country": source.get('Country'),  # Added Country
            "state": source.get('State'),  # Added State
            "display_address": source.get('Display_Address'),  # Added Display Address
            "image_url": source.get('Image_URL'),  # Added Image URL
            "alias": source.get('Alias'),  # Added Alias
            "review_count": source.get('Review_Count'),  # Added Review Count
            "price": source.get('Price'),  # Added Price
            "cuisine": source.get('Cuisine_Type'),
            "item": source.get('Item_Name'),
            "item_price": source.get('Item_Price'),  # Added Item Price
            "item_category": source.get('Item_Category'),  # Added Item Category
            "item_id": source.get('Item_id'),  
            "description": source.get('Description'),
            "rating": source.get("Rating"),
            "categories": source.get("Categories"),
            "primary_match": "semantic" if weighted_semantic > weighted_bm25 else "bm25",
            "metadata": {  # Additional metadata section
                "index": result['meta'].get('index'),
                "doc_type": result['meta'].get('doc_type'),
                "doc_id": result['meta'].get('doc_id', result['id'])
            }
        })
    
    if output:
        table_data = []
        headers = [
            "Rank", "Item Name", "Description", "Doc ID", 
            "Blended Score", "Semantic Score", "BM25 Score",
            "Match Type"
        ]
        
        for res in output:
            table_data.append([
                res['rank'],
                res['item'][:30] + '...' if len(res.get('item', '')) > 30 else res.get('item', 'N/A'),
                res["description"][:50] + '...' if len(res.get('description', '')) > 50 else res.get('description', 'N/A'),
                res["doc_id"],
                f"{res['blended_score']:.4f}",
                f"{res['normalized_semantic']:.4f} (orig: {res['original_semantic']:.2f})",
                f"{res['normalized_bm25']:.4f} (orig: {res['original_bm25']:.2f})",
                res['primary_match']
            ])
        
        print("\nSearch Results Table:")
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
        
        print(f"\nShowing top {len(output)} of {len(sorted_results)} total matches")
    
    return output




def generate_menu_item_response(claude_api_key, query, retrieved_chunks):
    restaurant_entries = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        entry_lines = []
        
        # Restaurant information with numbering
        entry_lines.append(f"### {idx}. Restaurant: {chunk.get('restaurant', 'Unknown')}")
        
        core_details = [
            ("Restaurant ID", chunk.get('restaurant_id')),
            ("Address", chunk.get('display_address')),
            ("City", chunk.get('city')),
            ("State", chunk.get('state', '').upper() if chunk.get('state') else None),
            ("Rating", f"{chunk.get('rating')}/5" if chunk.get('rating') else None)
        ]
        
        for label, value in core_details:
            if value:
                entry_lines.append(f"- **{label}**: {value}")
        
        entry_lines.append(f"### Menu Item: {chunk.get('item', 'N/A')}")
        
        item_details = [
            ("Item Category", chunk.get('item_category')),
            ("Item Price", chunk.get('item_price', 'Not available')),
            ("Description", chunk.get('description'))
        ]
        
        for label, value in item_details:
            if value:
                entry_lines.append(f"- **{label}**: {value}")
        
        additional_details = [
            ("ZIP Code", chunk.get('zip')),
            ("Country", chunk.get('country')),
            ("Yelp URL", chunk.get('yelp_url')),
            ("Image URL", chunk.get('image_url')),
            ("Alias", chunk.get('alias')),
            ("Review Count", chunk.get('review_count')),
            ("Price Range", chunk.get('price')),
            ("Cuisine", chunk.get('cuisine')),
            ("Categories", chunk.get('categories'))
        ]
        
        for label, value in additional_details:
            if value:
                entry_lines.append(f"- **{label}**: {value}")
        
        restaurant_entries.append("\n".join(entry_lines))

    system_prompt = (
        "You are a helpful restaurant concierge with expertise in menu items. Your job is to provide accurate, "
        "concise information about restaurant menu items based on the provided data. "
        "Follow these guidelines:\n"
        "1. Answer directly based ONLY on the provided restaurant and menu item information\n"
        "2. Highlight the most relevant details that match the user's query\n"
        "3. Organize information in a user-friendly, scannable format\n"
        "4. Include prices, descriptions, and restaurant location details when available\n"
        "5. Do not mention the technical aspects of how the information was retrieved\n"
        "6. If information is missing or limited, acknowledge this fact honestly\n"
        "7. Do not invent or assume details that are not in the provided data"
        "8. IMPORTANT: If you feel some menu item does not fit the query, feel free to OMIT it."
    
    )

    separator = "\n\n"
    prompt = (
        f"The user asked: \"{query}\"\n\n"
        "Below is information about relevant menu items at restaurants that match this query:\n\n"
        f"{separator.join(restaurant_entries)}\n\n"
        "Based on the information above, provide a helpful response that answers the user's query. "
        "Focus on the most relevant details, particularly the menu item itself, its price, description, "
        "and basic information about the restaurant where it's served. "
        "Format your response to be easy to read and understand. "
        "Ensure that each restaurant is numbered in the order they appear, like 1. Restaurant A, 2. Restaurant B, etc."
    )
    
    client = anthropic.Anthropic(api_key=claude_api_key)

    response = client.messages.create(
        model= "claude-3-haiku-20240307",
        max_tokens= 800,
        temperature= 0.3,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def identify_aggregation_and_generate_pandas_query(claude_api_key, query):
    
    system_prompt = """
    You are a specialized SQL and pandas analyzer focusing on identifying aggregation requirements in natural language queries.
    
    Your task is to determine whether a given query needs aggregation operations and to generate the appropriate pandas query.
    
    Parse the query carefully for indicators like:
    - Fastest growing, slowest growing, top, least, popular
    - Words suggesting grouping: "by", "per", "for each", "grouped by"
    - Aggregation keywords: "total", "average", "count", "sum", "minimum", "maximum"
    - Time-based grouping: "monthly", "yearly", "quarterly"
    
    Respond with a properly formatted Python dictionary literal that can be processed with ast.literal_eval().
    Do not use JSON formatting with newlines or escaped quotes.
    
    IMPORTANT: When requires_aggregation is False, the pandas_query MUST be an empty string.
    """
    categorical_columns = [
        "Quarter", "Year", "restaurant_id", "restaurant", "categories", 
        "city", "zip", "country", "state", "price", "cuisine", 
        "item_id", "item_category", "item"
    ]
    
    continuous_columns = [
        "rating", "review_count", "item_price"
    ]
    all_columns = categorical_columns + continuous_columns
    
    prompt = (
        "You are a SQL and pandas specialist. You need to identify if a natural language query requires aggregation "
        "and generate the appropriate pandas query.\n\n"
        f"Available columns in the dataset:\n"
        f"Categorical columns: {categorical_columns}\n"
        f"Continuous columns: {continuous_columns}\n\n"
        "Return your response as a Python dictionary literal that can be evaluated with ast.literal_eval().\n"
        "Use the following format for aggregation queries:\n"
        "{'requires_aggregation': True, 'group_by_columns': ['col1', 'col2'], "
        "'aggregate_columns': ['col3'], 'aggregate_functions': ['count', 'sum', 'mean', 'min', 'max'], "
        "'pandas_query': \"df.groupby(['col1', 'col2'])['col3'].agg(['count', 'sum']).reset_index()\"}\n\n"
        "For non-aggregation queries:\n"
        "{'requires_aggregation': False, 'pandas_query': ''}\n\n"
        "IMPORTANT: When requires_aggregation is False, the pandas_query MUST be an empty string.\n\n"
        "Make sure all quotes are properly escaped for ast.literal_eval() - use single quotes for the outer dictionary "
        "and double quotes for the pandas code strings when needed.\n\n"
        f"Query to analyze: {query}\n"
    )
    client = anthropic.Anthropic(api_key=claude_api_key)
    
    message = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    try:
        response_text = message.content[0].text.strip()
        response_text = response_text.replace('\n', '').replace('```python', '').replace('```', '')

        return response_text
    except Exception as e:
        return f"Error processing response: {str(e)}"



def generate_aggregation_response(claude_api_key, query, df):
    """
    Generates a user-friendly response using an LLM based on a query and a DataFrame containing restaurant and menu item data.

    Args:
        query (str): The user's query.
        df (pd.DataFrame): A DataFrame containing restaurant and menu item information.
        claude_api_key (str): API key for the Anthropic LLM.

    Returns:
        str: A formatted response generated by the LLM.
    """
    # Check if the DataFrame is empty
    if df.empty:
        return "No results found for your query."

    # Convert the DataFrame to a string format for the LLM
    df_str = df.to_string(index=False)

    # System prompt for the LLM
    system_prompt = (
        "You are a helpful restaurant concierge with expertise in menu items. Your job is to provide accurate, "
        "concise information about restaurant menu items based on the provided data. "
        "Follow these guidelines:\n"
        "1. Answer directly based ONLY on the provided restaurant and menu item information\n"
        "2. Highlight the most relevant details that match the user's query\n"
        "3. Organize information in a user-friendly, scannable format\n"
        "4. Include prices, descriptions, and restaurant location details when available\n"
        "5. Do not mention the technical aspects of how the information was retrieved\n"
        "6. If information is missing or limited, acknowledge this fact honestly\n"
        "7. Do not invent or assume details that are not in the provided data"
    )

    # User prompt for the LLM
    prompt = (
        f"The user asked: \"{query}\"\n\n"
        "Below is the data about relevant restaurants and menu items:\n\n"
        f"{df_str}\n\n"
        "Based on the information above, provide a helpful response that answers the user's query. "
        "Focus on the most relevant details, particularly the menu item itself, its price, description, "
        "and basic information about the restaurant where it's served. "
        "Format your response to be easy to read and understand."
    )

    client = anthropic.Anthropic(api_key=claude_api_key)

    # Generate the response using the LLM
    response = client.messages.create(
        model="claude-3-haiku-20240307",  # Use the appropriate model
        max_tokens=800,
        temperature=0.3,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text



def agentic_flow(claude_api_key, open_api_key, query, index_name, top_k, final_k, semantic_weight, bm25_weight, enable_hybrid, use_threshold=False, score_threshold=0.4):
    x = identify_aggregation_and_generate_pandas_query(claude_api_key, query)
    x = ast.literal_eval(x)
    
    if x["requires_aggregation"] is False:
        
        
        results = retrieve_chunks_hybrid(es, open_api_key,
                index_name=index_name,
                query_text=query,
                final_k=final_k,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight
            )
        
        return generate_menu_item_response(claude_api_key, query, results)
    
    else:
        print("AGG")
        results = retrieve_chunks_hybrid(es, open_api_key,
                index_name=index_name,
                query_text=query,
                final_k=final_k,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight
            )
        df = pd.DataFrame(results)
        df.columns = [col.lower() for col in df.columns]
        for col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
            except:
                continue
        print(x["pandas_query"])
        df = (eval(x["pandas_query"]))
        return generate_aggregation_response(claude_api_key, query, df )

import ast
import pandas as pd
import streamlit as st
from elasticsearch import Elasticsearch
from anthropic import Anthropic
import anthropic

# Include all your functions here: retrieve_chunks_hybrid, generate_menu_item_response, 
# identify_aggregation_and_generate_pandas_query, generate_aggregation_response, agentic_flow

def main():
    open_api_key = st.secrets["open_api_key"]
    elasticsearch_url = st.secrets["elasticsearch_url"]
    es_username = st.secrets["es_username"]
    es_password = st.secrets["es_password"]
    claude_api_key = st.secrets["claude_api_key"]

    # Initialize Elasticsearch client
    es = Elasticsearch(
        elasticsearch_url,
        basic_auth=(es_username, es_password),
        verify_certs=False,
        ssl_show_warn=False,
        request_timeout=60
    )

    index_name = "mocktail_index"

    # Sidebar controls
    with st.sidebar:
        st.header("Search Parameters")
        final_k = st.slider("Number of Results", 1, 250, 10)
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7)
        bm25_weight = 1.0 - semantic_weight

    st.text(f"Semantic Weight: {semantic_weight:.2f}, BM25 Weight: {bm25_weight:.2f}")

    # User input
    query = st.text_input("Enter your beverage query:", "What are some Sodas?")

    if st.button("Search"):
        with st.spinner("Searching across restaurants..."):
            # Call the agentic flow function
            llm_result = agentic_flow(
                claude_api_key=claude_api_key,
                open_api_key=open_api_key,
                query=query,
                index_name=index_name,
                top_k=100,
                final_k=final_k,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                enable_hybrid=True,
                use_threshold=True,
                score_threshold=0.3
            )


            # Display results
            if not llm_result:
                st.warning("No results found. Try adjusting your search parameters.")
            else:
                # Display results in an expandable section
                with st.expander(f"📊 See {len(results)} Search Results", expanded=True):
                    results_df = pd.DataFrame([{
                        "Item": r.get('item', 'N/A'),
                        "Restaurant": r.get('restaurant', 'N/A'),
                        "Description": (r.get('description', 'N/A')[:75] + '...') 
                        if len(r.get('description', '')) > 75 
                        else r.get('description', 'N/A'),
                        "Price": r.get('item_price', 'N/A'),
                        "Rating": r.get('rating', 'N/A'),
                        "Address": r.get('display_address', 'N/A'),
                        "Score": f"{r['blended_score']:.2f}",
                        "Match Type": r['primary_match']
                    } for r in results])

                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                format="%.2f",
                                min_value=0,
                                max_value=1.0
                            )
                        }
                    )

                # Display LLM-generated response
                st.text("Chatbot Output:")
                st.write(llm_result)


if __name__ == "__main__":
    main()
  
