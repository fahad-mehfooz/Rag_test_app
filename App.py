
import streamlit as st
import pandas as pd
import time
import openai
import elasticsearch
from elasticsearch import Elasticsearch
import time
from tabulate import tabulate
import polars as pl



def create_embeddings(text_chunks):
    
    openai.api_key = open_api_key

    batch_size = 2000  
    all_embeddings = []
    
    for i in range(0, len(text_chunks), batch_size):
        
        batch = text_chunks[i:i+batch_size]
        batch = [str(text) for text in batch if text]
        
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
                encoding_format="float"
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            time.sleep(0.5)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}")
            
        except Exception as e:
            print(f"Error in batch starting at index {i}: {e}")
    
    return all_embeddings


        
        



def retrieve_chunks_hybrid(index_name, query_text, top_k=100, final_k=10,
                            semantic_weight=0.7, bm25_weight=0.3,
                            enable_hybrid=True, use_threshold=False, score_threshold=0.5):
    """
    Retrieve chunks using hybrid search with both semantic and keyword components.
    
    Args:
        index_name (str): Name of the Elasticsearch index
        query_text (str): User query text
        top_k (int): Number of initial results to retrieve
        final_k (int): Number of final results to return
        semantic_weight (float): Weight for semantic search component (0-1)
        bm25_weight (float): Weight for BM25 search component (0-1)
        enable_hybrid (bool): Whether to enable hybrid search or use semantic only
        use_threshold (bool): Whether to filter by score threshold instead of top-k
        score_threshold (float): Minimum blended score threshold (0-1) when use_threshold=True
        
    Returns:
        list: List of retrieved chunks with scores and metadata including document ID
    """
    if not enable_hybrid:
        semantic_weight = 1.0
        bm25_weight = 0.0
        print("Hybrid search disabled. Using semantic search only.")
    
    if semantic_weight + bm25_weight != 1.0:
        total = semantic_weight + bm25_weight
        semantic_weight /= total
        bm25_weight /= total

    try:
        query_vector = create_embeddings([query_text])[0]
    except Exception as e:
        print(f"Embedding creation error: {e}")
        return []

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
    
    if use_threshold:
        filtered_results = [result for result in sorted_results if result['blended_score'] >= score_threshold]
        print(f"Threshold filtering: Found {len(filtered_results)} results with blended score >= {score_threshold}")
        selected_results = filtered_results
    else:
        selected_results = sorted_results[:final_k]
    
    output = []
    for i, result in enumerate(selected_results, 1):
        if not use_threshold and i > final_k:
            break
            
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
            "Rank", "Item Name", "Doc ID", 
            "Blended Score", "Semantic Score", "BM25 Score",
            "Match Type"
        ]
        
        for res in output:
            table_data.append([
                res['rank'],
                res['item'][:30] + '...' if len(res.get('item', '')) > 30 else res.get('item', 'N/A'),
                res["doc_id"],  # Use explicit doc_id field
                f"{res['blended_score']:.4f}",
                f"{res['normalized_semantic']:.4f} (orig: {res['original_semantic']:.2f})",
                f"{res['normalized_bm25']:.4f} (orig: {res['original_bm25']:.2f})",
                res['primary_match']
            ])
        
        print("\nSearch Results Table:")
        print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))
        
        if use_threshold:
            print(f"\nShowing {len(output)} results with blended score >= {score_threshold}")
        else:
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





def main():
    open_api_key = st.secrets["open_api_key"]
    elasticsearch_url = st.secrets["elasticsearch_url"]
    es_username = st.secrets["es_username"]
    es_password = st.secrets["es_password"]
    es_cloud_id = st.secrets["es_cloud_id"]
    
    
    openai.api_key = open_api_key
    
    auth = (es_username, es_password)
    
    es = Elasticsearch(
        elasticsearch_url,
        basic_auth=(es_username, es_password),
        verify_certs=False,  # Temporarily disable SSL verification
        ssl_show_warn=False,
        request_timeout=60
    )
    
    
    Mocktail_index_name = "mocktail_index"

    try:
        if es.ping():
            st.sidebar.success("✅ Connected to Elasticsearch")
        else:
            st.sidebar.error("❌ Could not connect to Elasticsearch")
    except Exception as es_conn_error:
        st.sidebar.error(f"Elasticsearch connection failed: {es_conn_error}")# Add index existence check
    if es.indices.exists(index=Mocktail_index_name):
        st.sidebar.success(f"✅ Index '{Mocktail_index_name}' exists")
    else:
        st.sidebar.error(f"❌ Index '{Mocktail_index_name}' not found")
        
    st.title("🍹 Menu Item Search Engine")
    st.markdown("Discover non-alcoholic beverage options across restaurants!")

    try:
        result = es.get(index=Mocktail_index_name, id="chunk_0")
        st.write("Elasticsearch Query Result:")
        st.json(result)  # Pretty-print JSON response
    except Exception as e:
        st.error(f"Error retrieving data: {e}")

    # Sidebar controls
    with st.sidebar:
        st.header("Search Parameters")
        final_k = st.slider("Number of Results", 1, 100, 10)
        semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7)
        bm25_weight = st.slider("BM25 Weight", 0.0, 1.0, 0.3)
        use_threshold = st.checkbox("Use Score Threshold")
        score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.4) if use_threshold else None

    # Main search interface
    query = st.text_input("Enter your beverage query:", "What are some Sodas?")

    if st.button("Search"):
        with st.spinner("Searching across restaurants..."):
            results = retrieve_chunks_hybrid(
                index_name=Mocktail_index_name,
                query_text=query,
                final_k=final_k,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                use_threshold=use_threshold,
                score_threshold=score_threshold
            )

        if not results:
            st.warning("No results found. Try adjusting your search parameters.")
            return

        # Display results in an expandable section
        with st.expander(f"📊 See {len(results)} Search Results", expanded=True):
            results_df = pd.DataFrame([{
                "Item": r.get('item', 'N/A'),
                "Restaurant": r.get('restaurant', 'N/A'),
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

if __name__ == "__main__":
    main()



