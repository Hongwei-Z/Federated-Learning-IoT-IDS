import pandas as pd
import numpy as np

def assign_random_client_id(csv_path: str, n_clients: int = 5) -> pd.DataFrame:
    """
    从csv加载数据后，随机为每行分配一个客户端ID (1 ~ n_clients)。
    返回带有新列 [Client_ID] 的DataFrame。
    """
    df = pd.read_csv(csv_path)
    
    # 为每行随机分配 [1, n_clients] 范围内的整数ID
    df["Client_ID"] = np.random.randint(1, n_clients + 1, size=len(df))
    
    return df

# 为数据添加Client_ID，并保存到新的CSV
df = assign_random_client_id("dataset.csv", n_clients=5)
df.to_csv("dataset_with_clients.csv", index=False)
print("随机客户端ID已写入 dataset_with_clients.csv！")
