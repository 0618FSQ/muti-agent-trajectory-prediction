from utils.visualization import ViszTool

from tqdm import tqdm



if __name__ == "__main__":
    
    csv_path = "/home/data1/prediction/dataset/argoverse/csv/train"
    
    visz = ViszTool(csv_path=csv_path)
    for i in tqdm(range(20, 100)):
        visz.visz(i)