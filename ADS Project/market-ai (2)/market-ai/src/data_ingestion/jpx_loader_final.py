cd ~/market-ai

# First, remove the broken file
rm src/data_ingestion/jpx_loader_final.py

# Now create the file properly (without any shell commands inside)
cat > src/data_ingestion/jpx_loader_final.py << 'EOF'
import pandas as pd
import cudf
import cupy as cp
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JPXProductionLoader:
    """
    Production-grade loader for JPX Tokyo Stock Exchange dataset
    Optimized for 24GB RAM + RTX 4050 GPU
    """
    
    def __init__(self, data_path="/mnt/c/Users/LOQ/OneDrive/Documents/ADS Project/jpx-tokyo-stock-exchange-prediction"):
        self.data_path = Path(data_path)
        self.train_path = self.data_path / "train_files"
        self.stock_list_path = self.data_path / "stock_list.csv"
        
        self.price_data = None
        self.stock_metadata = None
        
    def load_stock_metadata(self):
        """Load stock metadata from stock_list.csv"""
        logger.info("Loading stock metadata...")
        df = pd.read_csv(self.stock_list_path)
        relevant_cols = ['SecuritiesCode', 'Name', '33SectorName', '17SectorName', 'NewMarketSegment']
        available_cols = [col for col in relevant_cols if col in df.columns]
        df = df[available_cols]
        self.stock_metadata = cudf.from_pandas(df)
        logger.info(f"Loaded metadata for {len(self.stock_metadata)} stocks")
        return self.stock_metadata
    
    def load_price_data_chunked(self, nrows=None, chunksize=500000):
        """Load stock_prices.csv in chunks to GPU"""
        price_file = self.train_path / "stock_prices.csv"
        if not price_file.exists():
            logger.error(f"Price file not found: {price_file}")
            return None
        
        logger.info(f"Loading {price_file.name} ({price_file.stat().st_size / 1e9:.2f} GB)")
        sample = pd.read_csv(price_file, nrows=5)
        logger.info(f"Columns: {list(sample.columns)}")
        
        # Define columns to use
        use_cols = ['SecuritiesCode', 'Open', 'High', 'Low', 'Close', 'Volume']
        if 'Date' in sample.columns:
            use_cols.append('Date')
        
        # Read in chunks
        chunks = []
        total_rows = 0
        for i, chunk in enumerate(pd.read_csv(price_file, chunksize=chunksize, nrows=nrows,
                                              usecols=use_cols, low_memory=False)):
            gpu_chunk = cudf.from_pandas(chunk)
            chunks.append(gpu_chunk)
            total_rows += len(gpu_chunk)
            logger.info(f"  Chunk {i+1}: {len(gpu_chunk)} rows (Total: {total_rows:,})")
        
        if chunks:
            self.price_data = cudf.concat(chunks, ignore_index=True)
            logger.info(f"✅ Loaded {len(self.price_data)} rows into GPU memory")
            if 'Date' in self.price_data.columns:
                self.price_data['Date'] = cudf.to_datetime(self.price_data['Date'])
            return self.price_data
        else:
            logger.error("No data loaded")
            return None
    
    def preprocess_price_data(self):
        """Preprocess price data on GPU"""
        if self.price_data is None:
            logger.error("No price data loaded")
            return None
        
        logger.info("Preprocessing price data on GPU...")
        
        if 'Date' in self.price_data.columns:
            self.price_data = self.price_data.sort_values(['SecuritiesCode', 'Date'])
        
        if 'Close' in self.price_data.columns:
            # Shifted close
            self.price_data['Close_shift1'] = self.price_data.groupby('SecuritiesCode')['Close'].shift(1)
            # Simple returns
            self.price_data['Returns'] = (self.price_data['Close'] - self.price_data['Close_shift1']) / self.price_data['Close_shift1']
            # Log returns using cuDF's log (handles masks)
            close_clean = self.price_data['Close'].fillna(0)
            close_shift_clean = self.price_data['Close_shift1'].fillna(0)
            self.price_data['LogReturns'] = cudf.Series.log(close_clean) - cudf.Series.log(close_shift_clean)
            self.price_data['LogReturns'] = self.price_data['LogReturns'].replace([float('inf'), -float('inf')], 0)
            self.price_data = self.price_data.drop(columns=['Close_shift1'])
        
        if 'High' in self.price_data.columns and 'Low' in self.price_data.columns:
            self.price_data['Spread'] = (self.price_data['High'] - self.price_data['Low']) / self.price_data['Low']
            self.price_data['Spread'] = self.price_data['Spread'].fillna(0)
        
        if 'Returns' in self.price_data.columns:
            self.price_data['Returns'] = self.price_data['Returns'].fillna(0).replace([float('inf'), -float('inf')], 0)
            mean_ret = self.price_data['Returns'].mean()
            std_ret = self.price_data['Returns'].std()
            if std_ret > 0:
                self.price_data = self.price_data[abs(self.price_data['Returns'] - mean_ret) <= 5 * std_ret]
        
        logger.info(f"Preprocessing complete: {len(self.price_data)} rows remaining")
        return self.price_data
    
    def get_top_liquid_stocks(self, n=50):
        if self.price_data is None:
            self.load_price_data_chunked()
        if 'Volume' not in self.price_data.columns:
            all_stocks = self.price_data['SecuritiesCode'].unique()
            return all_stocks[:n].to_pandas().tolist()
        avg_volume = self.price_data.groupby('SecuritiesCode')['Volume'].mean()
        top_stocks = avg_volume.sort_values(ascending=False).head(n)
        return top_stocks.index.to_pandas().tolist()
    
    def create_panel_data(self, stocks=None, start_date=None, end_date=None):
        if self.price_data is None:
            self.load_price_data_chunked()
        if stocks is not None:
            mask = self.price_data['SecuritiesCode'].isin(stocks)
            filtered = self.price_data[mask]
        else:
            filtered = self.price_data
        if 'Date' in filtered.columns:
            if start_date:
                filtered = filtered[filtered['Date'] >= start_date]
            if end_date:
                filtered = filtered[filtered['Date'] <= end_date]
        if 'Close' in filtered.columns:
            pdf = filtered[['Date', 'SecuritiesCode', 'Close']].to_pandas()
            panel = pdf.pivot(index='Date', columns='SecuritiesCode', values='Close')
            panel = panel.fillna(method='ffill').fillna(method='bfill')
            panel_gpu = cudf.from_pandas(panel)
            logger.info(f"Created panel: {panel_gpu.shape[0]} dates × {panel_gpu.shape[1]} stocks")
            return panel_gpu
        return None
    
    def get_data_summary(self):
        summary = {
            'data_source': str(self.train_path),
            'price_file_exists': (self.train_path / "stock_prices.csv").exists(),
        }
        if self.price_data is not None:
            summary['total_rows'] = len(self.price_data)
            summary['unique_stocks'] = self.price_data['SecuritiesCode'].nunique() if 'SecuritiesCode' in self.price_data.columns else 0
            summary['columns'] = list(self.price_data.columns)
            if 'Date' in self.price_data.columns:
                summary['date_range'] = {
                    'start': str(self.price_data['Date'].min()),
                    'end': str(self.price_data['Date'].max())
                }
        if self.stock_metadata is not None:
            summary['stocks_with_metadata'] = len(self.stock_metadata)
        return summary

def quick_test():
    loader = JPXProductionLoader()
    print("="*60)
    print("JPX DATA LOADER TEST")
    print("="*60)
    
    print("\n1. Loading stock metadata...")
    metadata = loader.load_stock_metadata()
    if metadata is not None:
        print(f"   ✅ Loaded {len(metadata)} stocks")
    
    print("\n2. Loading price data (100k rows sample)...")
    price_data = loader.load_price_data_chunked(nrows=100000)
    if price_data is not None:
        print(f"   ✅ Loaded {len(price_data)} rows")
        print(f"   Columns: {list(price_data.columns)}")
        
        print("\n3. Preprocessing...")
        processed = loader.preprocess_price_data()
        print(f"   ✅ Preprocessed shape: {len(processed)}")
        
        print("\n4. Finding top liquid stocks...")
        top = loader.get_top_liquid_stocks(10)
        print(f"   ✅ Top 10: {top}")
        
        print("\n5. Data Summary:")
        summary = loader.get_data_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
    
    print("\n"+"="*60)
    print("✅ TEST COMPLETE!")
    return loader

if __name__ == "__main__":
    quick_test()
EOF