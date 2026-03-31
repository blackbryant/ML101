import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from clearml import Task, Dataset

def main():
    # ==========================================
    # 步驟 1：初始化 ClearML 任務
    # 這會在 ClearML 後台建立一個名為 "1. 資料取得" 的任務
    # ==========================================
    task = Task.init(project_name="Titanic_MLOps_Project", task_name="1. 資料取得")
    task.set_base_docker("python:3.9-slim")
    # 集中管理參數 (這些參數會自動顯示在 ClearML 後台，且可被遠端修改)
    args = {
        "competition_name": "titanic",
        "data_dir": "./data/titanic",
        "dataset_name": "Titanic Raw Dataset"
    }
    task.connect(args)

    print("🚀 開始執行鐵達尼號資料取得任務...")

    # ==========================================
    # 步驟 2：透過 Kaggle API 下載比賽資料
    # ==========================================
    os.makedirs(args["data_dir"], exist_ok=True)
    
    try:
        # 初始化並驗證 Kaggle API (會自動讀取環境變數)
        api = KaggleApi()
        api.authenticate()
        
        print(f"📦 正在從 Kaggle 下載 {args['competition_name']} 資料集...")
        api.competition_download_files(
            args["competition_name"], 
            path=args["data_dir"], 
            unzip=True
        )
        print("✅ 下載與解壓縮完成！")
    except Exception as e:
        print(f"❌ Kaggle 下載失敗，請檢查環境變數設定。錯誤訊息: {e}")
        return

    # ==========================================
    # 步驟 3：簡單驗證資料並記錄到 ClearML
    # ==========================================
    train_csv_path = os.path.join(args["data_dir"], "train.csv")
    if os.path.exists(train_csv_path):
        df = pd.read_csv(train_csv_path)
        print(f"\n📊 訓練集資料維度: {df.shape}")
        
        # 將 DataFrame 的前幾筆資料記錄到 ClearML 的 Console 中
        task.get_logger().report_text(f"訓練集預覽:\n{df.head().to_string()}")
    else:
        print("❌ 找不到 train.csv 檔案！")
        return

    # ==========================================
    # 步驟 4：建立 ClearML 資料集 (Data Versioning)
    # 這一步能確保未來的模型訓練使用的是鎖定版本的資料
    # ==========================================
    print("\n💾 正在將資料打包為 ClearML Dataset 版本...")
    try:
        dataset = Dataset.create(
            dataset_name=args["dataset_name"],
            dataset_project="Titanic_MLOps_Project",
            description="從 Kaggle 官方下載的鐵達尼號原始比賽資料 (train.csv, test.csv, gender_submission.csv)"
        )
        
        # 將下載的資料夾加入到這個資料集中
        dataset.add_files(path=args["data_dir"])
        
        # 上傳並定版 (Finalize)
        dataset.upload()
        dataset.finalize()
        print(f"✅ ClearML 資料集建立成功！版本 ID: {dataset.id}")
        
    except Exception as e:
        print(f"⚠️ 建立 ClearML Dataset 時發生錯誤: {e}")

    print("🎉 任務執行完畢！")

if __name__ == "__main__":
    main()
