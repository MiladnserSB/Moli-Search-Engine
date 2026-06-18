import os
import sys
import csv
import tempfile
import shutil

# =========================================================================
# 📂 1. Cache Redirection & Windows OS Protection
# =========================================================================
base_cache_dir = os.path.join(os.getcwd(), 'ir_datasets_cache')
local_tmp_dir = os.path.join(base_cache_dir, 'tmp')

os.makedirs(local_tmp_dir, exist_ok=True)

# Force ir_datasets and Python temp utilities to use local project directory
os.environ['IR_DATASETS_HOME'] = base_cache_dir
os.environ['TMPDIR'] = local_tmp_dir
os.environ['TEMP'] = local_tmp_dir
os.environ['TMP'] = local_tmp_dir
tempfile.tempdir = local_tmp_dir

# 🛠️ Fix System CSV field limit restriction for Windows environments
_original_field_size_limit = csv.field_size_limit
def _safe_field_size_limit(*args, **kwargs):
    if args and args[0] > 2147483647:
        args = (2147483647,) + args[1:]
    return _original_field_size_limit(*args, **kwargs)
csv.field_size_limit = _safe_field_size_limit
csv.field_size_limit(2147483647)
# =========================================================================

from scripts.database import init_database
from scripts.data_loader import IRDatasetDownloader

# ==========================================
# 🏁 Central Execution Engine (Main Entry)
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Starting Information Retrieval Dataset Pipeline...")
    print("=" * 60)

    # 1. Initialize DB Schema and Indexes
    init_database() 
    
    # 2. Target Repositories Configurations
    target_datasets = [
        {"name": "beir/quora/dev", "db_label": "quora_dev"},
        {"name": "lotte/lifestyle/dev/forum", "db_label": "lotte_lifestyle_dev"}
    ]
    
    # 3. Trigger Core Pipeline Downloader
    downloader = IRDatasetDownloader(target_datasets)
    downloader.execute_pipeline()
    
    # 4. Immediate Post-Processing Temporary Storage Purge
    print("\n🧹 Initiating storage cleanup operations...")
    try:
        if os.path.exists(local_tmp_dir):
            shutil.rmtree(local_tmp_dir)
            os.makedirs(local_tmp_dir, exist_ok=True)  # Recreate empty shell for subsequent stages
            print("✅ Storage Cache Purged: Temporary processing files successfully removed.")
    except Exception as e:
        print(f"⚠️ Cache Purge Warning: Some internal temp logs are locked by OS: {e}")

    print("\n🎉 [SUCCESS] All target datasets downloaded, parsed, and indexed in clean architecture!")
    print("=" * 60)