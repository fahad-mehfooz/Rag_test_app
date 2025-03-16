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

    # **Semantic Search**
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
            "restaurant_id": source.get('Restaurant_id'),
            "yelp_url": source.get('Yelp_URL'),
            "address": source.get('Address'),
            "city": source.get('City'),
            "zip": source.get('Zip'),
            "country": source.get('Country'),
            "state": source.get('State'),
            "display_address": source.get('Display_Address'),
            "image_url": source.get('Image_URL'),
            "alias": source.get('Alias'),
            "review_count": source.get('Review_Count'),
            "price": source.get('Price'),
            "cuisine": source.get('Cuisine_Type'),
            "item": source.get('Item_Name'),
            "item_price": source.get('Item_Price'),
            "item_category": source.get('Item_Category'),
            "description": source.get('Description'),
            "rating": source.get("Rating"),
            "categories": source.get("Categories"),
            "primary_match": "semantic" if weighted_semantic > weighted_bm25 else "bm25",
            "metadata": {
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


def group_and_aggregate(data, groupby_cols, agg_col, agg_func="count", top_n=None, sort_desc=True, filter_conditions=None):
    """
    Groups and aggregates a dataset with optional filtering using Polaris instead of Pandas.
    
    Parameters:
    - data (list of dicts): Raw input data to convert into DataFrame
    - groupby_cols (str or list): Column(s) to group by
    - agg_col (str): Column to perform aggregation on (e.g., "restaurant")
    - agg_func (str): Aggregation function ("count", "sum", "max", "min", etc.)
    - top_n (int, optional): If provided, returns only the top N results
    - sort_desc (bool): Whether to sort in descending order (default: True)
    - filter_conditions (dict, optional): Conditions to filter numeric columns, e.g., {"rating": (">=", 3)} or price categories
    
    Returns:
    - pl.DataFrame: Processed DataFrame with aggregated values, sorted
    """
    # Convert data to Polaris DataFrame
    df = pl.DataFrame(data)
    
    # Extract metadata fields if they exist
    if "metadata" in df.columns:
        df = df.with_columns([
            pl.col("metadata").struct.field("index").alias("metadata_index"),
            pl.col("metadata").struct.field("doc_id").alias("metadata_doc_id")
        ])
    
    # Apply filters if provided
    if filter_conditions:
        for col, condition in filter_conditions.items():
            if isinstance(condition, tuple):  # For numeric column conditions
                operator, value = condition
                if operator == ">=":
                    df = df.filter(pl.col(col) >= value)
                elif operator == "<=":
                    df = df.filter(pl.col(col) <= value)
                elif operator == ">":
                    df = df.filter(pl.col(col) > value)
                elif operator == "<":
                    df = df.filter(pl.col(col) < value)
                elif operator == "==":
                    df = df.filter(pl.col(col) == value)
                elif operator == "!=":
                    df = df.filter(pl.col(col) != value)
    
    agg_expr = None
    if agg_func == "count":
        agg_expr = pl.count(agg_col).alias(f"{agg_func}_{agg_col}")
    elif agg_func == "sum":
        agg_expr = pl.sum(agg_col).alias(f"{agg_func}_{agg_col}")
    elif agg_func == "max":
        agg_expr = pl.max(agg_col).alias(f"{agg_func}_{agg_col}")
    elif agg_func == "min":
        agg_expr = pl.min(agg_col).alias(f"{agg_func}_{agg_col}")
    elif agg_func == "mean" or agg_func == "avg":
        agg_expr = pl.mean(agg_col).alias(f"{agg_func}_{agg_col}")
    else:
        # For other aggregation functions, use a generic approach
        agg_expr = pl.col(agg_col).agg(agg_func).alias(f"{agg_func}_{agg_col}")
    
    # Convert groupby_cols to list if it's a string
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]
    
    # Perform groupby and aggregation
    sort_col = f"{agg_func}_{agg_col}"
    grouped_df = (
        df.group_by(groupby_cols)
        .agg([agg_expr])
        .sort(sort_col, descending=sort_desc)
    )
    
    if top_n is not None:
        grouped_df = grouped_df.limit(top_n)
    
    return grouped_df





import json
def generate_menu_item_response(query, retrieved_chunks, final_k):

    unique_entries = {}
    for chunk in retrieved_chunks:
        # Create a unique key based on restaurant ID and item name
        key = f"{chunk.get('restaurant_id', '')}-{chunk.get('item', '')}"
        # Only keep the first occurrence (presumably the most relevant)
        if key not in unique_entries and len(unique_entries) < final_k:
            unique_entries[key] = chunk
    
    restaurant_entries = []
    for chunk in unique_entries.values():
        entry_lines = []
        
        # Restaurant information
        entry_lines.append(f"### Restaurant: {chunk.get('restaurant', 'Unknown')}")
        
        # Core restaurant details
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
        
        # Additional restaurant details
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
    )

    separator = "\n\n"
    prompt = (
        f"The user asked: \"{query}\"\n\n"
        "Below is information about relevant menu items at restaurants that match this query:\n\n"
        f"{separator.join(restaurant_entries)}\n\n"
        "Based on the information above, provide a helpful response that answers the user's query. "
        "Focus on the most relevant details, particularly the menu item itself, its price, description, "
        "and basic information about the restaurant where it's served. "
        "Format your response to be easy to read and understand."
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
    
    return results.strip()


def main():
    open_api_key = st.secrets["open_api_key"]
    elasticsearch_url = st.secrets["elasticsearch_url"]
    es_username = st.secrets["es_username"]
    es_password = st.secrets["es_password"]
    es_cloud_id = st.secrets["es_cloud_id"]
    claude_api_key = st.secrets["claude_api_key"]

    
        
    auth = (es_username, es_password)
    
    es = Elasticsearch(
        elasticsearch_url,
        basic_auth=(es_username, es_password),
        verify_certs=False, 
        ssl_show_warn=False,
        request_timeout=60
    )
    
    
    index_name = "test_index"



    # Sidebar controls
    with st.sidebar:
        st.header("Search Parameters")
        final_k = st.slider("Number of Results", 1, 250, 10)
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7)
        bm25_weight = 1.0 - semantic_weight

    st.text(f"Semantic Weight: {semantic_weight:.2f}, BM25 Weight: {bm25_weight:.2f} ")
    


    query = st.text_input("Enter your beverage query:", "What are some Sodas?")

    if st.button("Search"):
        with st.spinner("Searching across restaurants..."):
            results = retrieve_chunks_hybrid(es, open_api_key,
                index_name=index_name,
                query_text=query,
                final_k=final_k,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight
            )

            llm_result = (generate_menu_item_response(query, results, final_k))
    
        if not results:
            st.warning("No results found. Try adjusting your search parameters.")
            return

        # Display results in an expandable section
        with st.expander(f"📊 See {len(results)} Search Results", expanded=True):
            results_df = pd.DataFrame([{
                "Item": r.get('item', 'N/A'),
                "Restaurant": r.get('restaurant', 'N/A'),
                "Description": (r.get('description', 'N/A') + '...') 
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

        
        st.text(f"Chatbot Output: ")
        st.text(llm_result)


if __name__ == "__main__":
    main()
