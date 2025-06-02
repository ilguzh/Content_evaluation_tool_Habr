import os
import pandas as pd
import json
import ast
from data_preprocessor import HabrDataPreprocessor
from typing import List, Optional, Union
import logging
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HabrDataProcessor:
    def __init__(self, data_dir: str = 'habr'):
        self.data_dir = data_dir
        self.preprocessor = HabrDataPreprocessor()
        
    # def get_data_files(self) -> List[str]:
    #     """Get list of all data files in the data directory."""
    #     return [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]

    def get_data_files(self) -> List[str]:
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        return all_files[:10]
    
    def parse_json_string(self, json_str: Union[str, list, dict]) -> Union[list, dict]:
        """Safely parse a JSON string that might be malformed."""
        try:
            if pd.isna(json_str):
                return []
            if isinstance(json_str, (list, dict)):
                return json_str
            # Handle case where string is already a JSON-formatted list
            if isinstance(json_str, str) and json_str.strip().startswith('['):
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        return ast.literal_eval(json_str)
                    except (ValueError, SyntaxError):
                        pass
            # Try to parse as regular string
            return [{'alias': json_str.strip()}] if json_str.strip() else []
        except Exception as e:
            logger.warning(f"Failed to parse JSON string: {str(json_str)[:100]}... Error: {str(e)}")
            return []
    
    def parse_tags(self, tags_str: Union[str, list]) -> List[str]:
        """Parse tags from comma-separated string."""
        try:
            if pd.isna(tags_str):
                return []
            if isinstance(tags_str, list):
                return tags_str
            # Remove any surrounding quotes and split by comma
            tags_str = str(tags_str).strip('"\'')
            return [tag.strip() for tag in tags_str.split(',')]
        except Exception as e:
            logger.warning(f"Error parsing tags: {str(e)}")
            return []
    
    def clean_html_text(self, html_text: str) -> str:
        """Extract clean text from HTML content."""
        try:
            if pd.isna(html_text):
                return ""
            soup = BeautifulSoup(str(html_text), 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logger.warning(f"Error cleaning HTML text: {str(e)}")
            return str(html_text)
    
    def read_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Read and preprocess a Habr CSV file."""
        try:
            # Read CSV with proper encoding
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Process JSON columns (hubs)
            json_columns = ['hubs_pro', 'hubs_nopro']
            for col in json_columns:
                if col in df.columns:
                    df[col] = df[col].apply(self.parse_json_string)
            
            # Process tags as comma-separated strings
            if 'tags' in df.columns:
                df['tags'] = df['tags'].apply(self.parse_tags)
            
            # Clean HTML text
            if 'text' in df.columns:
                df['text'] = df['text'].apply(self.clean_html_text)
            
            # Convert datetime
            if 'time_published' in df.columns:
                df['time_published'] = pd.to_datetime(df['time_published'])
            
            # Convert boolean - handle both string and numeric values
            if 'is_corporative' in df.columns:
                df['is_corporative'] = df['is_corporative'].apply(
                    lambda x: str(x).lower() in ['true', '1', 'yes'] if pd.notna(x) else False
                )
            
            # Convert numeric columns
            numeric_columns = [
                'bookmarks', 'comments_count', 'views', 'votes',
                'positive_votes', 'negative_votes', 'reading_time',
                'karma', 'karma_votes', 'rating'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Pre-process hubs and tags counts
            if 'hubs_pro' in df.columns:
                df['hubs_pro_count'] = df['hubs_pro'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            if 'hubs_nopro' in df.columns:
                df['hubs_nopro_count'] = df['hubs_nopro'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            if 'tags' in df.columns:
                df['tags_count'] = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            
            return df
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
            return None
    


    def extract_hub_aliases(self, hubs_data: Union[str, list, dict, np.ndarray]) -> str:
        """Extract hub aliases from JSON string, list, dict, or numpy array."""
        try:
            # Handle numpy arrays by converting to list
            if isinstance(hubs_data, np.ndarray):
                if hubs_data.size == 0:
                    return ""
                hubs_list = hubs_data.tolist()
            else:
                hubs_list = hubs_data

            # Handle None or NaN values
            if hubs_list is None or (isinstance(hubs_list, float) and np.isnan(hubs_list)):
                return ""

            # Parse string input into Python objects
            if isinstance(hubs_list, str):
                try:
                    hubs_list = ast.literal_eval(hubs_list)
                except (ValueError, SyntaxError, json.JSONDecodeError):
                    # Treat as plain alias string if parsing fails
                    return hubs_list.strip()

            aliases = []
            # Process list or tuple of hubs
            if isinstance(hubs_list, (list, tuple)):
                for item in hubs_list:
                    if isinstance(item, dict):
                        alias = item.get('alias', '')
                        if alias:
                            aliases.append(alias)
                    elif isinstance(item, str):
                        aliases.append(item.strip())
            # Process single hub dictionary
            elif isinstance(hubs_list, dict):
                alias = hubs_list.get('alias', '')
                if alias:
                    aliases.append(alias)
            # Handle unexpected types (e.g., numbers)
            else:
                return str(hubs_list).strip()

            return ','.join(aliases) if aliases else ""
        except Exception as e:
            logger.warning(f"Error extracting hub aliases: {str(e)}")
            return ""
    
    def process_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Process a single data file."""
        try:
            logger.info(f"Processing file: {file_path}")
            full_path = os.path.join(self.data_dir, file_path)
            
            # Read and preprocess the file
            df = self.read_file(full_path)
            if df is None:
                logger.error(f"Failed to read {file_path}")
                return None
            
            # Process hub data - keep only aliases in original columns
            if 'hubs_pro' in df.columns:
                logger.info("Processing profile hubs...")
                df['hubs_pro'] = df['hubs_pro'].apply(self.extract_hub_aliases)
            
            if 'hubs_nopro' in df.columns:
                logger.info("Processing additional hubs...")
                df['hubs_nopro'] = df['hubs_nopro'].apply(self.extract_hub_aliases)
            
            # Combine hubs_pro and hubs_nopro into a single hubs column
            if 'hubs_pro' in df.columns and 'hubs_nopro' in df.columns:
                logger.info("Combining hub columns...")
                df['hubs'] = df.apply(lambda row: 
                    f"{row['hubs_pro']},{row['hubs_nopro']}".strip(','), 
                    axis=1
                )
                # Remove the original hub columns
                df = df.drop(columns=['hubs_pro', 'hubs_nopro'])
            
            # Add derived features
            if all(col in df.columns for col in ['views', 'comments_count']):
                df['comments_per_view'] = df['comments_count'] / (df['views'] + 1)
            
            if all(col in df.columns for col in ['positive_votes', 'negative_votes']):
                total_votes = df['positive_votes'] + df['negative_votes']
                df['vote_ratio'] = df['positive_votes'] / (total_votes + 1)
            
            if all(col in df.columns for col in ['text', 'reading_time']):
                df['text_length'] = df['text'].str.len()
                df['reading_speed'] = df['text_length'] / (df['reading_time'] * 60 + 1)
            
            # Remove hub count columns if they exist
            count_columns = ['hubs_count', 'hubs_pro_count', 'hubs_nopro_count', 'hubs_additional_count']
            for col in count_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            logger.info(f"Successfully processed {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_all_files(self) -> pd.DataFrame:
        """Process all data files and combine them into a single DataFrame."""
        data_files = self.get_data_files()
        if not data_files:
            raise ValueError(f"No data files found in {self.data_dir}")
            
        logger.info(f"Found {len(data_files)} files to process")
        
        # Process each file and combine results
        all_data = []
        for file in data_files:
            processed_df = self.process_file(file)
            if processed_df is not None:
                all_data.append(processed_df)
        
        if not all_data:
            raise ValueError("No data was successfully processed")
            
        # Combine all processed DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset contains {len(combined_df)} rows")
        
        return combined_df
    
    def save_processed_data(self, df: pd.DataFrame, output_file: str = 'processed_habr_data6.parquet', last_6_months: bool = False):
        """Save the processed data to a Parquet file with compression. Optionally filter to last 6 months."""
        try:
            # Ensure the DataFrame is not empty
            if df.empty:
                raise ValueError("Cannot save empty DataFrame")

            # Optionally filter to last 6 months
            if last_6_months and 'time_published' in df.columns:
                df['time_published'] = pd.to_datetime(df['time_published'], errors='coerce')
                cutoff = pd.Timestamp.now() - pd.DateOffset(months=6)
                df = df[df['time_published'] >= cutoff]
                output_file = 'processed_habr_data_last_6_months.parquet'

            # Convert datetime columns to string format for better Parquet compatibility
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                df[col] = df[col].astype(str)

            # Ensure all columns are compatible with Parquet
            for col in df.columns:
                if df[col].dtype == 'object':
                    if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                        df[col] = df[col].astype(str)

            df.to_parquet(
                output_file,
                engine='pyarrow',
                compression='gzip',
                index=False
            )
            if not os.path.exists(output_file):
                raise FileNotFoundError(f"Parquet file was not created at {output_file}")
            try:
                pd.read_parquet(output_file)
            except Exception as e:
                raise ValueError(f"Created Parquet file is invalid: {str(e)}")
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            logger.info(f"Successfully saved processed data to {output_file}")
            logger.info(f"File size: {file_size_mb:.2f} MB")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

def main(output_last_6_months: bool = True):
    processor = HabrDataProcessor()
    try:
        processed_df = processor.process_all_files()
        processor.save_processed_data(processed_df, last_6_months=output_last_6_months)
        print("\nProcessing Summary:")
        print(f"Total rows processed: {len(processed_df)}")
        print(f"Number of columns: {len(processed_df.columns)}")
        print("\nColumns:")
        print(processed_df.columns.tolist())
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")

if __name__ == "__main__":
    # Set to True to output only last 6 months of data
    main(output_last_6_months=True) 